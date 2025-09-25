"""
main.py

This module defines API endpoints for cluster search and classification.
Endpoints:
    - POST /enrich_embedding: Enrich document embeddings by processing uploaded JSONL file.
    - POST /run_clustering: Perform macro and micro clustering on document embeddings.
    - POST /text_represent_micro: Summarize and label microclusters using LLM
    - POST /text_represent_macro: Summarize and label macroclusters using LLM
    - GET /search: Search documents based on query embedding and return top results.
"""

import math
from xml.parsers.expat import errors
from fastapi import FastAPI, UploadFile, File, Query, Body, HTTPException
from fastapi.encoders import jsonable_encoder
from typing import Any, List, Optional, Dict
from fastapi.responses import JSONResponse
import json, io, uuid, datetime, psycopg2, torch
import psycopg2, pandas as pd, numpy as np, ast, json
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from psycopg2.extras import Json, execute_values
import spacy
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os
import umap
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from supabase import create_client, Client
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")


# Load environment variables
load_dotenv()

# ==========================
# Config
# ==========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT_MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

# Supabase connection details
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
SUPABASE_DB_HOST = os.getenv("SUPABASE_DB_HOST")
SUPABASE_DB_NAME = os.getenv("SUPABASE_DB_NAME")
SUPABASE_DB_USER = os.getenv("SUPABASE_DB_USER")
SUPABASE_DB_PASS = os.getenv("SUPABASE_DB_PASS")
SUPABASE_DB_PORT = os.getenv("SUPABASE_DB_PORT")
EXPO_PUBLIC_SUPABASE_URL = os.getenv("EXPO_PUBLIC_SUPABASE_URL")
SUPABASE_DB_KEY = os.getenv("SUPABASE_DB_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Tables in Supabase:
DATA_TEXT_TABLE = "public.data_text"
DATA_VECTOR_TABLE = "vecs.data_vector"
CLUSTER_ASSIGNMENTS_TABLE = "public.cluster_assignments"
CLUSTER_MACRO_TABLE = "cluster_macro2"
CLUSTER_MACRO_TABLE_PUBLIC = "public.cluster_macro2"
CLUSTER_MICRO_TABLE = "cluster_micro2"
CLUSTER_MICRO_TABLE_PUBLIC = "public.cluster_micro2"

# ==========================
# Init models & DB
# ==========================

# Load embedding model
print("🔹 Loading embedding model...")
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)
model.eval()

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Supabase client
supabase: Client = create_client(EXPO_PUBLIC_SUPABASE_URL, SUPABASE_DB_KEY)

conn = psycopg2.connect(
    host=SUPABASE_DB_HOST,
    dbname=SUPABASE_DB_NAME,
    user=SUPABASE_DB_USER,
    password=SUPABASE_DB_PASS,
    port=SUPABASE_DB_PORT,
    sslmode="require"
)

engine = create_engine(SUPABASE_DB_URL)

cur = conn.cursor()
print("✅ Connected to Supabase")

# FastAPI app
app = FastAPI()

# ==========================
# Helper functions
# ==========================

# ---------- Get Embedding ----------
def get_embedding(text: str):
    """
    Get embedding for a given text using the loaded model

    Args:
        text (str): The input text to embed.

    Returns:
        np.ndarray: The embedding vector for the input text.
    """
    if not text or not text.strip():
        return None
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb.astype(np.float32)

# ---------- Preprocess JSON ----------
def preprocess_json(json_data):
    """
    Preprocess JSON metadata by filtering out low-information lines.

    Args:
        json_data (dict): The input JSON data containing metadata.
    
    Returns:
        dict: The preprocessed JSON data with filtered metadata.
    """
    spacy_model = spacy.load("es_core_news_sm")

    def is_low_info(line):
        """
        Determine if a line is low-information based on length and stopword ratio.
        """
        doc = spacy_model(line)
        if len(doc) < 3:
            return True
        stopword_ratio = sum(token.is_stop for token in doc) / len(doc)
        if stopword_ratio > 0.7:
            return True
        return False

    def preprocess(text):
        """
        Preprocess text by lowercasing, removing stopwords, and lemmatizing.
        """
        doc = spacy_model(text.lower())
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and token.is_alpha
        ]
        return " ".join(tokens)

    # Filter and preprocess headings
    for tag in ["h1", "h2", "h3"]:
        if tag in json_data["metadata"]:
            filtered = [line for line in json_data["metadata"][tag] if not is_low_info(line)]
            json_data["metadata"][tag] = [preprocess(text) for text in filtered]

    return json_data

# ---------- Generate Parsing ----------
def generate_parsing(json_data, openai_api_key=OPENAI_API_KEY, model_name=GPT_MODEL_NAME):
    """
    Generate structured parsing of a document using LLM.

    Args:
        json_data (dict): The input JSON data containing metadata.
        openai_api_key (str): The API key for OpenAI.
        model_name (str): The name of the LLM model to use.
    
    Returns:    
        tuple: A tuple containing the structured output dictionary and token usage statistics.

    """
    json_data = preprocess_json(json_data)  # still filter out noise

    client = OpenAI(api_key=openai_api_key)

    # Collect headings
    h1 = "\n".join(json_data["metadata"].get("h1", []))
    h2 = "\n".join(json_data["metadata"].get("h2", []))
    h3 = "\n".join(json_data["metadata"].get("h3", []))
    headings_text = f"{h1}\n\n{h2}\n\n{h3}".strip()

    # Collect image alt text
    alt_texts = json_data["metadata"].get("alt", [])
    alt_text_joined = "\n".join(alt_texts)

    # Send raw headings + alt text to LLM
    prompt_content = f"""
You are an expert in multilingual SEO semantic analysis.

Given the following website content:

### Headings (H1, H2, H3)
{headings_text}

### Image Descriptions (alt text)
{alt_text_joined}

Please extract and return ONLY a valid JSON object with this structure:
{{
  "entities": {{
    "label_name": ["entity1", "entity2"],  // each label contains a list of unique entities with no duplicates
    ...
  }},
  "suggested_title": "string",
  "faq_pairs": [
    {{"question": "string", "answer": "string"}}
  ],
  "user_search_intent": "string"
}}

Rules:
- Group entities by label, storing all unique entity texts in an array for each label.
- The "entities" object should include ALL significant entities found in headings or alt text, grouped under their label.
- No duplicates in each label's array.
- Limit the FAQ list to a maximum of 3 pairs that are most relevant to the page.
- Do not add explanations or extra commentary outside the JSON.
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a multilingual SEO and semantic parsing assistant."},
            {"role": "user", "content": prompt_content}
        ],
        temperature=0.3,
        max_tokens=1200
    )

    llm_json = json.loads(response.choices[0].message.content)

    # Limit FAQ pairs to maximum 3
    faq_pairs = llm_json.get("faq_pairs", [])[:3]

    neat_output = {
        "index": json_data.get("index"),
        "url": json_data.get("url"),
        "metadata": json_data.get("metadata"),
        "entities": llm_json.get("entities", {}),
        "suggested_title": llm_json.get("suggested_title"),
        "faq_pairs": faq_pairs,
        "user_search_intent": llm_json.get("user_search_intent"),
        # "total_tokens_used": response.usage.prompt_tokens + response.usage.completion_tokens
    }

    return neat_output, response.usage

# ---------- Parse Embedding ----------
def parse_embedding(x):
    """
    Parse embedding from string to numpy array.

    Args:
        x (str): The input string representation of the embedding.
    
    Returns:
        np.ndarray: The parsed embedding as a numpy array.
    """
    if x is None:
        return np.zeros(768, dtype=np.float32)
    try:
        return np.array(ast.literal_eval(x), dtype=np.float32)
    except Exception:
        return np.zeros(768, dtype=np.float32)
    

# ---------- LLM Call ----------
def run_llm(prompt: str) -> dict:
    """
    Generate a response from the LLM based on the provided prompt.

    Args:
        prompt (str): The input prompt for the LLM.

    Returns:
        dict: The JSON response from the LLM or the raw response if parsing fails.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    raw_text = response.choices[0].message.content.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {"raw": raw_text}
    

# ==========================
# API Endpoints
# ==========================

# ========== Stage 1 Endpoint: Enrich & Embedding ==========
@app.post("/enrich_embedding")
async def enrich_embedding(file: UploadFile = File(...)):
    """
    - Enrich document embeddings by processing uploaded JSONL file. 
    - Enriched data is stored in Supabase tables: data_text and data_vector.

    - **Args**:
        - file (UploadFile): The uploaded JSONL file containing documents to process.

    - **Returns**:
        - JSONResponse: A response indicating the success or failure of the operation.
    """

    # Load JSONL from uploaded file
    def load_jsonl_from_upload(file: UploadFile):
        """
        Load and parse JSONL file from upload

        Args:
            file (UploadFile): The uploaded JSONL file.
        
        Returns:
            list: A list of parsed JSON objects.
        """

        content = file.file.read().decode("utf-8").strip().splitlines()
        return [json.loads(line) for line in content if line.strip()]
    
    # Save document and its embedding to Supabase
    def save_document_and_vector(record):
        """
        Save a single document and its embeddings to Supabase

        Args:
            record (dict): The document record to save.

        Returns:
            str: The ID of the saved document.
        """

        doc_id = str(uuid.uuid4())
        now = datetime.datetime.utcnow()

        cur.execute(f"""
            INSERT INTO {DATA_TEXT_TABLE} (
                id, url, suggested_title, user_search_intent,
                metadata_h1, metadata_h2, metadata_h3, faq_pairs, entities, created_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            doc_id,
            record.get("url"),
            record.get("suggested_title"),
            record.get("user_search_intent"),
            record.get("metadata", {}).get("h1", []),
            record.get("metadata", {}).get("h2", []),
            record.get("metadata", {}).get("h3", []),
            Json(record.get("faq_pairs", [])),
            Json(record.get("entities", {})),
            now 
        ))

        title_vec = get_embedding(record.get("suggested_title", ""))
        intent_vec = get_embedding(record.get("user_search_intent", ""))
        metadata_parts = []
        for lvl in ["h1", "h2", "h3"]:
            metadata_parts.extend(record.get("metadata", {}).get(lvl, []))
        metadata_vec = get_embedding(" ".join(metadata_parts))
        faq_vec = get_embedding(" ".join([f"Q:{p['question']} A:{p['answer']}" for p in record.get("faq_pairs", [])]))
        entities_vec = get_embedding(" ".join([f"{k}: {v}" for k, v in record.get("entities", {}).items()]))

        cur.execute(f"""
            INSERT INTO {DATA_VECTOR_TABLE} (
                id, suggested_title, metadata, user_search_intent, faq_pairs, entities
            ) VALUES (%s,%s,%s,%s,%s,%s)
        """, (
            doc_id,
            title_vec.tolist() if title_vec is not None else None,
            metadata_vec.tolist() if metadata_vec is not None else None,
            intent_vec.tolist() if intent_vec is not None else None,
            faq_vec.tolist() if faq_vec is not None else None,
            entities_vec.tolist() if entities_vec is not None else None
        ))

        conn.commit()
        return doc_id

    try:
        records = load_jsonl_from_upload(file)
        saved_ids = []
        for rec in records:
            # print("📌 Record:", rec)
            parsed, response_usage = generate_parsing(rec, OPENAI_API_KEY, GPT_MODEL_NAME)
            print("✅ Parsed")
            doc_id = save_document_and_vector(parsed)
            saved_ids.append(doc_id)

        return JSONResponse(content={
            "status": "success",
            "rows_inserted": len(saved_ids),
            "document_ids": saved_ids,
            "supabase_tables": [DATA_TEXT_TABLE, DATA_VECTOR_TABLE]
        })
    except Exception as e:
        conn.rollback()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


# ========== Stage 2 Endpoint ==========
@app.post("/run_clustering")
async def run_clustering():
    """
    - Run clustering on document embeddings.
    - Performs macro and micro clustering using UMAP and HDBSCAN.
    - Data is retrieved from Supabase tables: data_text, data_vector.
    - Saves results to Supabase tables: cluster_macro, cluster_micro, cluster_assignments.

    - **Args**:
        None

    - **Returns**:
        JSONResponse: A response indicating the success or failure of the operation, including counts of clusters and assignments.
    """
    
    # Run micro clustering on a subset
    def run_micro_clustering(subset, min_cluster_size=5, n_neighbors=10, n_components=10):
        """
        Run UMAP + HDBSCAN on a subset and return labels + reduced embeddings

        Args:
            subset (pd.DataFrame): The subset of data to cluster.
            min_cluster_size (int): Minimum cluster size for HDBSCAN.
            n_neighbors (int): Number of neighbors for UMAP.
            n_components (int): Number of components for UMAP.

        Returns:
            tuple: A tuple containing micro labels and UMAP reduced embeddings.
        """

        X_sub = np.vstack(subset["combined_embedding"].values)

        n_comp = min(n_components, X_sub.shape[0] - 1)
        n_neigh = min(n_neighbors, X_sub.shape[0] - 1)

        umap_sub = umap.UMAP(
            n_neighbors=n_neigh,
            n_components=n_comp,
            metric="cosine",
            random_state=42
        )
        X_sub_umap = umap_sub.fit_transform(X_sub)

        micro_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom"
        )
        micro_labels = micro_clusterer.fit_predict(X_sub_umap)
        return micro_labels, X_sub_umap

    try:
        # 1. Fetch data
        # Tables: data_text, data_vector
        query = f"""
        SELECT v.id,
               v.suggested_title AS suggested_title_embedding,
               v.metadata AS metadata_embedding,
               v.user_search_intent AS user_search_intent_embedding,
               v.faq_pairs AS faq_pairs_embedding,
               v.entities AS entities_embedding,
               d.url,
               d.suggested_title AS suggested_title_text,
               d.user_search_intent AS user_search_intent_text,
               d.metadata_h1,
               d.metadata_h2,
               d.metadata_h3,
               d.faq_pairs AS faq_pairs_text,
               d.entities AS entities_text
        FROM {DATA_VECTOR_TABLE} v
        JOIN {DATA_TEXT_TABLE} d ON d.id = v.id
        """

        # Fetch data
        df = pd.read_sql(query, engine)
        if df.empty:
            return JSONResponse(content={"status": "error", "message": "No data found"}, status_code=400)

        # 2. Convert embeddings
        for col in ["suggested_title_embedding", "metadata_embedding",
                    "user_search_intent_embedding", "faq_pairs_embedding",
                    "entities_embedding"]:
            df[col] = df[col].apply(parse_embedding)

        df["combined_embedding"] = df.apply(
            lambda row: np.mean([
                row["suggested_title_embedding"],
                row["metadata_embedding"],
                row["user_search_intent_embedding"],
                row["faq_pairs_embedding"],
                row["entities_embedding"]
            ], axis=0),
            axis=1
        )

        # 3. Macro clustering
        X = np.vstack(df['combined_embedding'].values)

        umap_model = umap.UMAP(
            n_neighbors=15,
            n_components=50,
            metric='cosine',
            random_state=42
        )
        X_umap = umap_model.fit_transform(X)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=30,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        macro_labels = clusterer.fit_predict(X_umap)
        df['macro_cluster_id'] = macro_labels

        # Evaluate Macro Clusters
        mask = macro_labels != -1
        if mask.sum() > 1 and len(set(macro_labels[mask])) > 1:
            sil = silhouette_score(X_umap[mask], macro_labels[mask])
            dbi = davies_bouldin_score(X_umap[mask], macro_labels[mask])
            chi = calinski_harabasz_score(X_umap[mask], macro_labels[mask])
            print(f"[Macro] Silhouette: {sil:.4f}, Davies-Bouldin: {dbi:.4f}, Calinski-Harabasz: {chi:.4f}")
        else:
            print("[Macro] Not enough clusters for evaluation")

        print(f"[Macro] Number of clusters: {len(set(macro_labels)) - (1 if -1 in macro_labels else 0)}")
        print(f"[Macro] Number of noise points: {sum(macro_labels == -1)}")

        # 4. Micro clustering
        df["micro_cluster_id"] = -1
        sil_threshold = 0.6

        for cluster_id in set(macro_labels):
            if cluster_id == -1:
                continue
            subset = df[df["macro_cluster_id"] == cluster_id]
            if len(subset) < 10:
                continue

            # First attempt with default params
            micro_labels, X_sub_umap = run_micro_clustering(subset, min_cluster_size=3, n_neighbors=5, n_components=5)
            df.loc[subset.index, "micro_cluster_id"] = micro_labels

            # Evaluate
            mask = micro_labels != -1
            if mask.sum() > 1 and len(set(micro_labels[mask])) > 1:
                sil = silhouette_score(X_sub_umap[mask], micro_labels[mask])
                dbi = davies_bouldin_score(X_sub_umap[mask], micro_labels[mask])
                chi = calinski_harabasz_score(X_sub_umap[mask], micro_labels[mask])
                print(f"[Micro | Macro {cluster_id}] Silhouette: {sil:.4f}, DBI: {dbi:.4f}, CHI: {chi:.4f}")

                # 🔄 Retry if silhouette too low
                if sil < sil_threshold and sil != -1:
                    print(f"[Recluster] Macro {cluster_id} → retry with different hyperparameters...")
                    micro_labels, X_sub_umap = run_micro_clustering(subset, min_cluster_size=2, n_neighbors=5, n_components=1)
                    df.loc[subset.index, "micro_cluster_id"] = micro_labels

                    # Re-evaluate
                    mask = micro_labels != -1
                    if mask.sum() > 1 and len(set(micro_labels[mask])) > 1:
                        sil = silhouette_score(X_sub_umap[mask], micro_labels[mask])
                        dbi = davies_bouldin_score(X_sub_umap[mask], micro_labels[mask])
                        chi = calinski_harabasz_score(X_sub_umap[mask], micro_labels[mask])
                        print(f"[Re-Micro | Macro {cluster_id}] Silhouette: {sil:.4f}, DBI: {dbi:.4f}, CHI: {chi:.4f}")
                    else:
                        print(f"[Re-Micro | Macro {cluster_id}] Still not enough clusters.")
            else:
                print(f"[Micro | Macro {cluster_id}] Not enough clusters for evaluation")

        # 5. Prepare metadata
        macro_info, micro_info, assignments = [], [], []
        for cluster_id in set(macro_labels):
            if cluster_id == -1:
                continue
            cluster_embeddings = np.vstack(df[df['macro_cluster_id'] == cluster_id]['combined_embedding'].values)
            centroid = cluster_embeddings.mean(axis=0).tolist()

            macro_info.append({
                "cluster_id": int(cluster_id),
                "count": len(cluster_embeddings),
                "cluster_name": "", # empty for now
                "representative_text": "", # empty for now
                "representative_keywords": "", # empty for now
                "centroid_embedding": json.dumps(centroid)
            })

            subset = df[df["macro_cluster_id"] == cluster_id]
            for mc_id in set(subset["micro_cluster_id"]):
                if mc_id == -1:
                    continue
                mc_embeddings = np.vstack(subset[subset["micro_cluster_id"] == mc_id]['combined_embedding'].values)
                mc_centroid = mc_embeddings.mean(axis=0).tolist()
                micro_info.append({
                    "cluster_id": int(cluster_id),
                    "microcluster_id": int(mc_id),
                    "microcluster_name": "",
                    "count": len(mc_embeddings),
                    "representative_text": "",
                    "representative_keywords": "",
                    "centroid_embedding": json.dumps(mc_centroid)
                })

        for _, row in df.iterrows():
            assignments.append({
                "doc_id": str(row["id"]),
                "cluster_id": int(row["macro_cluster_id"]),
                "microcluster_id": int(row["micro_cluster_id"]),
                "probability": 1.0
            })

        # Re-connect to ensure fresh connection
        conn = psycopg2.connect(
            host=SUPABASE_DB_HOST,
            dbname=SUPABASE_DB_NAME,
            user=SUPABASE_DB_USER,
            password=SUPABASE_DB_PASS,
            port=SUPABASE_DB_PORT,
            sslmode="require"
        )

        cur = conn.cursor()
        print("✅ Connected to Supabase")

        print("\n🚀 Inserting results into Supabase...")
        # 6. Save results to Supabase
        cur.execute(f"DROP TABLE IF EXISTS {CLUSTER_ASSIGNMENTS_TABLE}")
        cur.execute(f"DROP TABLE IF EXISTS {CLUSTER_MACRO_TABLE}")
        cur.execute(f"DROP TABLE IF EXISTS {CLUSTER_MICRO_TABLE}")

        cur.execute(f"""
            CREATE TABLE {CLUSTER_ASSIGNMENTS_TABLE} (
                doc_id UUID,
                cluster_id INT,
                microcluster_id INT,
                probability FLOAT
            )
        """)
        cur.execute(f"""
            CREATE TABLE {CLUSTER_MACRO_TABLE_PUBLIC} (
                cluster_id INT PRIMARY KEY,
                count INT,
                cluster_name TEXT,
                representative_text TEXT,
                representative_keywords TEXT,
                centroid_embedding JSON
            )
        """)
        cur.execute(f"""
            CREATE TABLE {CLUSTER_MICRO_TABLE_PUBLIC} (
                cluster_id INT,
                microcluster_id INT,
                microcluster_name TEXT,
                count INT,
                representative_text TEXT,
                representative_keywords TEXT,
                centroid_embedding JSON
            )
        """)

        execute_values(cur, f"""
            INSERT INTO {CLUSTER_MACRO_TABLE_PUBLIC} (cluster_id, count, cluster_name, representative_text, representative_keywords, centroid_embedding)
            VALUES %s
        """, [(row["cluster_id"], row["count"], row["cluster_name"], row["representative_text"],
               row["representative_keywords"], row["centroid_embedding"]) for row in macro_info])

        execute_values(cur, f"""
            INSERT INTO {CLUSTER_MICRO_TABLE_PUBLIC} (cluster_id, microcluster_id, microcluster_name, count, representative_text, representative_keywords, centroid_embedding)
            VALUES %s
        """, [(row["cluster_id"], row["microcluster_id"], row["microcluster_name"], row["count"],
               row["representative_text"], row["representative_keywords"], row["centroid_embedding"]) for row in micro_info])

        execute_values(cur, f"""
            INSERT INTO {CLUSTER_ASSIGNMENTS_TABLE} (doc_id, cluster_id, microcluster_id, probability)
            VALUES %s
        """, [(row["doc_id"], row["cluster_id"], row["microcluster_id"], row["probability"]) for row in assignments])

        conn.commit()

        return JSONResponse(content={
            "status": "success",
            "macro_clusters": len(macro_info),
            "micro_clusters": len(micro_info),
            "assignments": len(assignments)
        })


    except Exception as e:
        conn.rollback()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


# ========== Stage 3 Endpoint : Microcluster ========== 
@app.post("/text_represent_micro")
async def text_represent_micro():
    """
    - Represent a microcluster of documents.
    - Summarizes and labels microclusters using LLM.
    - Data is retrieved from Supabase tables: data_text, cluster_micro, cluster_assignments.

    - **Args**:
        None

    - **Returns**:
        JSONResponse: A response indicating the success or failure of the operation, including counts of processed microclusters.
    """
    
    # Summarization prompt builders and logic
    def build_micro_prompt(df_chunk: pd.DataFrame, cluster_id: int, micro_id: Any) -> str:
        """
        Build a prompt for summarizing a microcluster.
        
        Args:
            df_chunk (pd.DataFrame): The subset of documents in the microcluster.
            cluster_id (int): The macro cluster ID.
            micro_id (Any): The micro cluster ID.
        
        Returns:
            str: The constructed prompt string.
        """

        docs = []
        for _, row in df_chunk.iterrows():
            title = row.get('suggested_title_text', '') or ""
            intent = row.get('user_search_intent_text', '') or ""
            entities = row.get('entities_text', '') or ""
            doc_str = f"- Title: {title}\n  Intent: {intent}\n  Entities: {entities}\n"
            docs.append(doc_str)
        joined_docs = "\n".join(docs)

        return f"""
    You are analyzing a microcluster of documents. Each document contains several fields
    (title, search intent, entities, etc.).

    Cluster ID: {cluster_id}, Microcluster ID: {micro_id}

    Here are the documents from this microcluster:
    ---
    {joined_docs}
    ---

    Please create a structured JSON object with the following fields:

    {{
    "summary": "A 2–3 sentence description of the main theme of this microcluster.",
    "key_points": ["3–6 bullet points highlighting key recurring ideas or topics."],
    "keywords": ["10–20 representative keywords or phrases (short, lowercase)."],
    "microcluster_name": "A short 2–5 word descriptive label for this microcluster."
    }}

    Rules:
    - Do not invent facts not present in the documents.
    - Use clear, concise English.
    - Keywords should reflect actual terms from the text.
    - The microcluster name should be broad enough to describe the set, but specific enough to distinguish it.
    """.strip()

    # Summarize a microcluster subset
    def summarize_microcluster(df_subset: pd.DataFrame, cluster_id: int, micro_id: Any,
                           batch_size: int = 25) -> Dict[str, Any]:
        """
        Summarize a microcluster based on its size. 
        Handles small, medium, and large microclusters differently.

        Args:
            df_subset (pd.DataFrame): The subset of documents in the microcluster.
            cluster_id (int): The macro cluster ID.
            micro_id (Any): The micro cluster ID.
            batch_size (int): The batch size for large microclusters.

        Returns:
            dict: A dictionary containing the mode, prompts, and info about the summarization.  
        """

        result = {"mode": None, "prompts": [], "info": {}}
        n = len(df_subset)

        if n == 0:
            result["mode"] = "empty"
            result["info"] = {"n_docs": 0}
            return result

        df_local = df_subset.copy().reset_index(drop=False)
        df_local["__emb"] = df_local["combined_embedding"].apply(lambda x: np.asarray(x, dtype=float))
        result["info"]["n_docs"] = n

        if n < 10:
            result["mode"] = "small"
            result["prompts"].append({
                "prompt_id": f"{micro_id}_all",
                "prompt": build_micro_prompt(df_local, cluster_id, micro_id),
                "doc_indices": df_local["index"].tolist(),
                "doc_ids": df_local["id"].tolist()
            })
            result["info"]["selection"] = "all_docs"

        elif n <= 30:
            result["mode"] = "medium"
            emb_stack = np.vstack(df_local["__emb"].values)
            centroid = emb_stack.mean(axis=0)
            df_local["__dist"] = df_local["__emb"].apply(lambda v: np.linalg.norm(v - centroid))
            top_df = df_local.nsmallest(10, "__dist")
            result["prompts"].append({
                "prompt_id": f"{micro_id}_top10",
                "prompt": build_micro_prompt(top_df, cluster_id, micro_id),
                "doc_indices": top_df["index"].tolist(),
                "doc_ids": top_df["id"].tolist()
            })
            result["info"]["selection"] = "topk_centroid"

        else:
            result["mode"] = "large"
            batches = np.array_split(df_local, math.ceil(n / batch_size))
            for idx, batch in enumerate(batches, start=1):
                result["prompts"].append({
                    "prompt_id": f"{micro_id}_part{idx}",
                    "prompt": build_micro_prompt(batch, cluster_id, f"{micro_id}_part{idx}"),
                    "doc_indices": batch["index"].tolist(),
                    "doc_ids": batch["id"].tolist()
                })
            result["info"]["selection"] = "batches"

        return result

    # Save microcluster summary back to Supabase
    def save_microcluster_result(cluster_id, microcluster_id, llm_result: dict):
        """
        Save the LLM-generated summary and keywords for a microcluster back to Supabase.

        Args:
            cluster_id (int): The macro cluster ID.
            microcluster_id (Any): The micro cluster ID.
            llm_result (dict): The LLM-generated result containing summary, key points, keywords, and microcluster name.
        """
        summary = llm_result.get("summary", "")
        key_points = llm_result.get("key_points", [])
        keywords = llm_result.get("keywords", [])
        micro_name = llm_result.get("microcluster_name", "")

        if isinstance(key_points, list):
            key_points = "; ".join([str(k) for k in key_points])
        if isinstance(keywords, list):
            keywords = ", ".join([str(k) for k in keywords])

        representative_text = summary
        if key_points:
            representative_text += "\n\n- " + "\n- ".join(key_points.split("; "))

        data = {
            "cluster_id": int(cluster_id) if cluster_id is not None else None,
            "microcluster_id": str(microcluster_id) if microcluster_id is not None else None,
            "microcluster_name": str(micro_name) if micro_name is not None else None,
            "representative_text": str(representative_text) if representative_text is not None else None,
            "representative_keywords": str(keywords) if keywords is not None else None,
        }

        result = supabase.table(CLUSTER_MICRO_TABLE) \
                        .update(data) \
                        .eq("cluster_id", int(cluster_id)) \
                        .eq("microcluster_id", str(microcluster_id)) \
                        .execute()
        print(f"✅ Saved cluster_id: {int(cluster_id)}, microcluster_id: {str(microcluster_id)} to Supabase")

    try:
        # 1. Fetch data from Supabase
        # Tables: cluster_assignments, data_text, data_vector
        sql_all = f"""
        SELECT
        ca.doc_id AS id,
        ca.cluster_id,
        ca.microcluster_id,
        d.suggested_title AS suggested_title_text,
        d.user_search_intent AS user_search_intent_text,
        d.entities AS entities_text,
        v.suggested_title AS suggested_title_embedding,
        v.metadata AS metadata_embedding,
        v.user_search_intent AS user_search_intent_embedding,
        v.faq_pairs AS faq_pairs_embedding,
        v.entities AS entities_embedding
        FROM {CLUSTER_ASSIGNMENTS_TABLE} ca
        JOIN {DATA_TEXT_TABLE} d ON d.id = ca.doc_id
        JOIN {DATA_VECTOR_TABLE} v ON v.id = ca.doc_id
        """
        
        df = pd.read_sql(sql_all, engine)

        print(f"✅ Fetched {len(df)} rows from Supabase")

        # 2. Preprocess embeddings
        for col in [
            "suggested_title_embedding",
            "metadata_embedding",
            "user_search_intent_embedding",
            "faq_pairs_embedding",
            "entities_embedding",
        ]:
            df[col] = df[col].apply(parse_embedding)

        df["combined_embedding"] = df.apply(
            lambda r: np.mean(
                [
                    r["suggested_title_embedding"],
                    r["metadata_embedding"],
                    r["user_search_intent_embedding"],
                    r["faq_pairs_embedding"],
                    r["entities_embedding"],
                ],
                axis=0,
            ),
            axis=1,
        )

        # Handle None in text fields
        df["entities_text"] = df["entities_text"].apply(lambda x: str(x) if x is not None else "")

        print("🔹 Preview:")
        print(df.head())

        # 3. Summarize each microcluster
        prompts = []

        for (cluster_id, micro_id), df_subset in df.groupby(["cluster_id", "microcluster_id"]):
            results = summarize_microcluster(df_subset, cluster_id, micro_id, batch_size=25)
            first_prompt = results["prompts"][0] if results["prompts"] else None

            if not first_prompt:
                continue

            llm_result = run_llm(first_prompt["prompt"])
            save_microcluster_result(cluster_id, micro_id, llm_result)

    except Exception as e:
        conn.rollback()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


# # ========== Stage 3 Endpoint : Macrocluster ==========
@app.post("/text_represent_macro")
async def text_represent_macro():
    """
    - Represent a macrocluster of documents.
    - Summarizes and labels macroclusters using LLM.
    - Data is retrieved from Supabase tables: cluster_macro, cluster_micro.

    - **Args**:
        None

    - **Returns**:
        JSONResponse: A response indicating the success or failure of the operation, including counts of processed macroclusters.
    """

    # Fetch microclusters for a given cluster_id
    def get_microclusters_for_cluster(cluster_id: int):
        """
        Fetch microclusters for a given cluster_id from Supabase.

        Args:
            cluster_id (int): The macro cluster ID.

        Returns:
            list: A list of microclusters with their details.
        """

        response = supabase.table(CLUSTER_MICRO_TABLE).select(
            "microcluster_id, microcluster_name, representative_text, representative_keywords"
        ).eq("cluster_id", cluster_id).execute()

        if not response.data:
            print(f"⚠️ No microclusters found for cluster_id {cluster_id}")
            return []

        print(f"✅ Retrieved {len(response.data)} microclusters for cluster {cluster_id}")
        return response.data

    # Build macro prompt
    def build_macro_prompt(cluster_id: int, microclusters: list, batch_id=None) -> str:
        """
        Build a prompt for summarizing a macrocluster from its microclusters.
        Args:
            cluster_id (int): The macro cluster ID.
            microclusters (list): The list of microclusters to include in the prompt.
            batch_id (Optional[int]): The batch ID, if applicable.

        Returns:
            str: The constructed prompt string.
        """
        
        parts = []
        for m in microclusters:
            name = m.get("microcluster_name", "")
            text = m.get("representative_text", "")
            keywords = m.get("representative_keywords", "")
            parts.append(
                f"- Microcluster {m['microcluster_id']} ({name}):\n"
                f"  Summary: {text}\n"
                f"  Keywords: {keywords}\n"
            )

        joined = "\n\n".join(parts)

        prompt = f"""
    You are analyzing a macrocluster of documents.
    This macrocluster contains several microclusters, each with a summary and keywords.

    Cluster ID: {cluster_id}{f", Batch {batch_id}" if batch_id else ""}

    Here are the microclusters:
    ---
    {joined}
    ---

    Please create a structured JSON object with the following fields:

    {{
    "summary": "A 2–3 sentence summary of the main theme of this batch of microclusters.",
    "key_points": ["3–6 bullet points highlighting recurring themes."],
    "keywords": ["10–20 representative keywords or phrases (short, lowercase)."],
    "batch_name": "A short 2–5 word descriptive label for this batch."
    }}

    Rules:
    - Do not invent facts not present in the microclusters.
    - Use clear, concise English.
    - Keywords should be actual recurring terms across microclusters.
    - The batch_name should be broad but specific enough to describe the set.
    """.strip()

        return prompt

    # Summarize a macrocluster in batches
    def build_reduce_prompt(cluster_id: int, batch_results: list) -> str:
        """
        Build a prompt for reducing the results of multiple batches into a cohesive summary.
        
        Args:
            cluster_id (int): The macro cluster ID. 
            batch_results (list): The list of batch results to include in the prompt.

        Returns:
            str: The constructed prompt string.
        """
        parts = []
        for i, br in enumerate(batch_results, start=1):
            summary = br["llm_result"].get("summary", "")
            key_points = br["llm_result"].get("key_points", [])
            keywords = br["llm_result"].get("keywords", [])
            batch_name = br["llm_result"].get("batch_name", "")
            parts.append(
                f"- Batch {i} ({batch_name}):\n"
                f"  Summary: {summary}\n"
                f"  Key Points: {key_points}\n"
                f"  Keywords: {keywords}\n"
            )

        joined = "\n\n".join(parts)

        prompt = f"""
    You are analyzing a macrocluster of documents.

    Cluster ID: {cluster_id}

    You are given summaries of several batches of microclusters.
    Each batch has its own summary, key points, and keywords.

    Here are the batch-level summaries:
    ---
    {joined}
    ---

    Please create a structured JSON object with the following fields:

    {{
    "summary": "A 3–5 sentence integrated summary of the entire macrocluster.",
    "key_points": ["5–8 key points synthesizing the recurring ideas across batches."],
    "keywords": ["15–25 representative keywords or phrases (short, lowercase)."],
    "cluster_name": "A short 2–5 word descriptive label for the entire macrocluster."
    }}

    Rules:
    - Focus on themes that recur across multiple batches.
    - Do not repeat batch-specific details unless central to the overall cluster.
    - Keywords should represent the macrocluster as a whole.
    - The cluster_name should be broad but distinctive.
    """.strip()

        return prompt
    

    # Summarize macrocluster
    def save_macrocluster_result(cluster_id: int, final_result: dict):
        """
        Save the final result of a macrocluster to the database.

        Args:
            cluster_id (int): The macro cluster ID.
            final_result (dict): The final LLM-generated result containing summary, key points, keywords, and cluster name.
        
        """
        summary = final_result.get("summary", "")
        key_points = final_result.get("key_points", [])
        keywords = final_result.get("keywords", [])

        # Handle cluster_name correctly
        cluster_name = final_result.get("cluster_name")
        if not cluster_name:
            cluster_name = final_result.get("batch_name", f"Cluster {cluster_id}")  # ✅ fallback

        # Representative text
        representative_text = summary
        if isinstance(key_points, list) and key_points:
            representative_text += "\n\n- " + "\n- ".join(key_points)

        # Keywords format
        if isinstance(keywords, list):
            keywords_str = ", ".join(keywords)
        else:
            keywords_str = str(keywords)

        data = {
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,  # always filled now ✅
            "representative_text": representative_text,
            "representative_keywords": keywords_str,
        }

        result = supabase.table(CLUSTER_MACRO_TABLE).upsert(data).execute()
        print("✅ Saved to Supabase:", result)

    try:
        # 1. Fetch macroclusters
        response = supabase.table(CLUSTER_MACRO_TABLE).select("cluster_id").execute()
        if not response.data:
            return JSONResponse(content={"status": "error", "message": "No macroclusters found"}, status_code=400)

        cluster_ids = [row["cluster_id"] for row in response.data]
        print(f"🔹 Found {len(cluster_ids)} macroclusters")

        # 2. Process each macrocluster
        for cluster_id in cluster_ids:
            microclusters = get_microclusters_for_cluster(cluster_id)
            if not microclusters:
                continue

            n = len(microclusters)
            print(f"\n🚀 Processing Macrocluster {cluster_id} with {n} microclusters")

            batch_size = 25
            batch_results = []

            if n <= batch_size:
                prompt = build_macro_prompt(cluster_id, microclusters)
                llm_result = run_llm(prompt)
                batch_results.append({
                    "batch_id": 1,
                    "llm_result": llm_result
                })
            else:
                batches = np.array_split(microclusters, math.ceil(n / batch_size))
                for idx, batch in enumerate(batches, start=1):
                    prompt = build_macro_prompt(cluster_id, batch, batch_id=idx)
                    llm_result = run_llm(prompt)
                    batch_results.append({
                        "batch_id": idx,
                        "llm_result": llm_result
                    })

            # Reduce step
            if len(batch_results) == 1:
                final_result = batch_results[0]["llm_result"]
            else:
                reduce_prompt = build_reduce_prompt(cluster_id, batch_results)
                final_result = run_llm(reduce_prompt)

            save_macrocluster_result(cluster_id, final_result)

        return JSONResponse(content={"status": "success", "processed_clusters": len(cluster_ids)})
    
    except Exception as e:
        conn.rollback()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


# Endpoint: New url
@app.post("/search_cluster")
async def search_cluster(file: UploadFile = File(...)):
    """
    - Search for a cluster in the database.
    - Accepts a JSON or JSONL file containing one or more records.
    - Each record should have fields like suggested_title, user_search_intent, metadata, faq_pairs, entities.
    - Returns the assigned macro and micro clusters, along with related documents.

    - **Args**:
        - file (UploadFile): The uploaded JSON or JSONL file.

    - **Returns**:
        - JSONResponse: A response containing the classification results for each record.
    """

    # Load json
    def load_json(uploaded_file: UploadFile):
        text = uploaded_file.file.read().decode("utf-8")
        if uploaded_file.filename.endswith(".jsonl"):
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        elif uploaded_file.filename.endswith(".json"):
            return json.loads(text)
        else:
            raise ValueError("Unsupported file type. Must be .json or .jsonl")

    # Embedding and classification
    def embed_record(record: dict) -> np.ndarray:
        """
        Create a combined embedding for the record using multiple fields.
         1. suggested_title
         2. user_search_intent 
         3. metadata (h1, h2, h3)
         4. faq_pairs
         5. entities

        Returns:
            np.ndarray: The combined embedding vector.
        """
        title_vec = get_embedding(record.get("suggested_title", ""))
        intent_vec = get_embedding(record.get("user_search_intent", ""))

        metadata_vec = get_embedding(" ".join(
            sum([record.get("metadata", {}).get(lvl, []) for lvl in ["h1","h2","h3"]], [])
        ))

        faq_vec = get_embedding(" ".join(
            [f"Q:{p.get('question','')} A:{p.get('answer','')}" for p in record.get("faq_pairs", [])]
        ))

        entities_text = " ".join([f"{k}:{v}" for k, v in record.get("entities", {}).items()])
        entities_vec = get_embedding(entities_text)

        return np.mean([title_vec, intent_vec, metadata_vec, faq_vec, entities_vec], axis=0)

    # Classify record
    def classify_record(record: dict, micro_df, macro_df, assign_df, top_k_related=5):
        """
        Classify a single record into macro and micro clusters.
        1. Embed the record
        2. Find closest macro cluster
        3. Find closest micro cluster within that macro
        4. Find related documents in the same macro cluster
        5. Return structured result

        Args:
            record (dict): The input record to classify.
            micro_df (pd.DataFrame): DataFrame of micro clusters.
            macro_df (pd.DataFrame): DataFrame of macro clusters.
            assign_df (pd.DataFrame): DataFrame of document assignments to clusters.
            top_k_related (int): Number of related documents to return.

        Returns:
            dict: The classification result with assigned clusters and related documents.
        """
        emb = embed_record(record).reshape(1, -1)

        # Macro
        macro_embs = np.vstack(macro_df["centroid_embedding"].values)
        sim_macro = cosine_similarity(emb, macro_embs)[0]
        macro_info = macro_df.iloc[int(np.argmax(sim_macro))].to_dict()

        # Micro
        micro_subset = micro_df[micro_df["cluster_id"] == macro_info["cluster_id"]]
        micro_info = {}
        if len(micro_subset) > 0:
            micro_embs = np.vstack(micro_subset["centroid_embedding"].values)
            sim_micro = cosine_similarity(emb, micro_embs)[0]
            micro_info = micro_subset.iloc[int(np.argmax(sim_micro))].to_dict()

        # Related docs
        docs_in_macro = assign_df[assign_df["cluster_id"] == macro_info["cluster_id"]].copy()
        related = []
        if len(docs_in_macro) > 0:
            sims = []
            for _, row in docs_in_macro.iterrows():
                doc_vec = get_embedding(row["suggested_title"] or "")
                sims.append(cosine_similarity(emb, doc_vec.reshape(1, -1))[0][0])
            docs_in_macro["similarity"] = sims
            related = docs_in_macro.sort_values("similarity", ascending=False).head(top_k_related)
            related = related[["doc_id","url","suggested_title","user_search_intent","entities","cluster_id","microcluster_id","similarity"]] \
                            .to_dict(orient="records")

        return {
            **record,
            "assigned_macro": {
                "cluster_id": macro_info["cluster_id"],
                "cluster_name": macro_info["cluster_name"],
                "keywords": macro_info["representative_keywords"],
                "count": macro_info["count"],
            },
            "assigned_micro": {
                "microcluster_id": micro_info.get("microcluster_id"),
                "microcluster_name": micro_info.get("microcluster_name"),
                "keywords": micro_info.get("representative_keywords"),
                "count": micro_info.get("count"),
            } if micro_info else None,
            "related_docs": related
        }
    

    try:
        # Load cluster data from Supabase
        print("🔄 Connecting to Supabase...")
        macro_df = pd.read_sql(f"SELECT * FROM {CLUSTER_MACRO_TABLE_PUBLIC}", engine)
        micro_df = pd.read_sql(f"SELECT * FROM {CLUSTER_MICRO_TABLE_PUBLIC}", engine)
        assign_df = pd.read_sql(f"""
            SELECT a.doc_id, a.cluster_id, a.microcluster_id, d.url, 
                   d.suggested_title, d.user_search_intent, d.entities
            FROM {CLUSTER_ASSIGNMENTS_TABLE} a
            JOIN {DATA_TEXT_TABLE} d ON a.doc_id = d.id
        """, engine)

        # Convert embeddings
        macro_df["centroid_embedding"] = macro_df["centroid_embedding"].apply(parse_embedding)
        micro_df["centroid_embedding"] = micro_df["centroid_embedding"].apply(parse_embedding)
        print("✅ Cluster metadata loaded")

        # 1. Load input record(s)
        records = load_json(file)
        if isinstance(records, dict):
            records = [records]
        print(f"✅ Loaded {len(records)} record(s)")

        all_results = []
        errors = []

        # 2. Process each record
        for i, rec in enumerate(records, start=1):
            print(f"\n--- Processing record {i}/{len(records)} ---")
            try:
                # Parse the record
                parsed, usage = generate_parsing(rec, OPENAI_API_KEY, GPT_MODEL_NAME)
                # Classify
                classified = classify_record(parsed, micro_df, macro_df, assign_df, top_k_related=5)
                # Append result
                all_results.append(classified)
                print(f"✅ Processed record {i}/{len(records)}")
            except Exception as e:
                errors.append({"record": i, "error": str(e)})

        # 3. Return structured response
        return JSONResponse(
            status_code=200,
            content=jsonable_encoder({
                "status": "ok",
                "processed": len(all_results),
                "errors": errors,
                "results": all_results,
            })
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


