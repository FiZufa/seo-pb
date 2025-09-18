from http.client import HTTPException
import math
from fastapi import FastAPI, UploadFile, File, Query
from typing import Any, List, Optional
from fastapi.responses import JSONResponse
import json, io, uuid, datetime, psycopg2, torch
import psycopg2, pandas as pd, numpy as np, ast, json
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
from psycopg2.extras import Json, execute_values
import spacy
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

import umap
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from supabase import create_client, Client

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

SUPABASE_DB_KEY = os.getenv("SUPABASE_DB_KEY")

DATABASE_URL = os.getenv("DATABASE_URL")

# ==========================
# Init models & DB
# ==========================

# Load embedding sebaiknya sebelum atau sesudah connect ke db?

# Load embedding model
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)
model.eval()

client = OpenAI(api_key=OPENAI_API_KEY)

# supabase: Client = create_client(SUPABASE_DB_URL, SUPABASE_DB_KEY)

conn = psycopg2.connect(
    host=SUPABASE_DB_HOST,
    dbname=SUPABASE_DB_NAME,
    user=SUPABASE_DB_USER,
    password=SUPABASE_DB_PASS,
    port=SUPABASE_DB_PORT,
    sslmode="require"
)

# conn = psycopg2.connect(DATABASE_URL, sslmode="require")

engine = create_engine(SUPABASE_DB_URL)

cur = conn.cursor()
print("✅ Connected to Supabase")

# ==========================
# FastAPI app
# ==========================
app = FastAPI()

# ==========================
# Helper functions
# ==========================
def load_jsonl_from_upload(file: UploadFile):
    content = file.file.read().decode("utf-8").strip().splitlines()
    return [json.loads(line) for line in content if line.strip()]

def get_embedding(text: str):
    if not text or not text.strip():
        return None
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb.astype(np.float32)

# ---------- Preprocess JSON ----------
def preprocess_json(json_data):
    spacy_model = spacy.load("es_core_news_sm")

    def is_low_info(line):
        doc = spacy_model(line)
        if len(doc) < 3:
            return True
        stopword_ratio = sum(token.is_stop for token in doc) / len(doc)
        if stopword_ratio > 0.7:
            return True
        return False

    def preprocess(text):
        doc = spacy_model(text.lower())
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and token.is_alpha
        ]
        return " ".join(tokens)

    for tag in ["h1", "h2", "h3"]:
        if tag in json_data["metadata"]:
            filtered = [line for line in json_data["metadata"][tag] if not is_low_info(line)]
            json_data["metadata"][tag] = [preprocess(text) for text in filtered]

    return json_data

# ---------- Generate Parsing ----------
def generate_parsing(json_data, openai_api_key, model_name="gpt-4o-mini"):
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

def save_document_and_vector(record):
    doc_id = str(uuid.uuid4())
    now = datetime.datetime.utcnow()

    cur.execute("""
        INSERT INTO public.data_text (
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

    cur.execute("""
        INSERT INTO vecs.data_vector (
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


def parse_embedding(x):
    if x is None:
        return np.zeros(768, dtype=np.float32)
    try:
        return np.array(ast.literal_eval(x), dtype=np.float32)
    except Exception:
        return np.zeros(768, dtype=np.float32)
    

def run_micro_clustering(subset, min_cluster_size=5, n_neighbors=10, n_components=10):
    """Run UMAP + HDBSCAN on a subset and return labels + reduced embeddings"""
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

def run_llm(prompt: str) -> dict:
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
    


# ========== Stage 1 Endpoint ==========
@app.post("/enrich_embedding")
async def enrich_embedding(file: UploadFile = File(...)):
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
            "supabase_tables": ["public.data_text", "vecs.data_vector"]
        })
    except Exception as e:
        conn.rollback()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


# ========== Stage 2 Endpoint ==========
@app.post("/run_clustering")
async def run_clustering():
    try:
        # 1. Fetch data
        query = """
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
        FROM vecs.data_vector v
        JOIN public.data_text d ON d.id = v.id
        """
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

        # AttributeError: module 'umap' has no attribute 'UMAP'
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

        # ✅ Evaluate Macro Clusters
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
        cur.execute("DROP TABLE IF EXISTS cluster_assignments_2")
        cur.execute("DROP TABLE IF EXISTS cluster_macro_2")
        cur.execute("DROP TABLE IF EXISTS cluster_micro_2")

        cur.execute("""
            CREATE TABLE cluster_assignments_2 (
                doc_id UUID,
                cluster_id INT,
                microcluster_id INT,
                probability FLOAT
            )
        """)
        cur.execute("""
            CREATE TABLE cluster_macro_2 (
                cluster_id INT PRIMARY KEY,
                count INT,
                cluster_name TEXT,
                representative_text TEXT,
                representative_keywords TEXT,
                centroid_embedding JSON
            )
        """)
        cur.execute("""
            CREATE TABLE cluster_micro_2 (
                cluster_id INT,
                microcluster_id INT,
                microcluster_name TEXT,
                count INT,
                representative_text TEXT,
                representative_keywords TEXT,
                centroid_embedding JSON
            )
        """)

        execute_values(cur, """
            INSERT INTO cluster_macro_2 (cluster_id, count, cluster_name, representative_text, representative_keywords, centroid_embedding)
            VALUES %s
        """, [(row["cluster_id"], row["count"], row["cluster_name"], row["representative_text"],
               row["representative_keywords"], row["centroid_embedding"]) for row in macro_info])

        execute_values(cur, """
            INSERT INTO cluster_micro_2 (cluster_id, microcluster_id, microcluster_name, count, representative_text, representative_keywords, centroid_embedding)
            VALUES %s
        """, [(row["cluster_id"], row["microcluster_id"], row["microcluster_name"], row["count"],
               row["representative_text"], row["representative_keywords"], row["centroid_embedding"]) for row in micro_info])

        execute_values(cur, """
            INSERT INTO cluster_assignments_2 (doc_id, cluster_id, microcluster_id, probability)
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
# class MicroclusterRequest(BaseModel):
#     target_cluster_id: int = 7
#     target_microcluster_id: int = 17  
    
# @app.post("/summarize_microcluster")
# async def summarize_microcluster(request: MicroclusterRequest):

#     def build_micro_prompt(df_chunk: pd.DataFrame, cluster_id: int, micro_id: Any) -> str:
#         docs = []
#         for _, row in df_chunk.iterrows():
#             title = row.get('suggested_title_text', '') or ""
#             intent = row.get('user_search_intent_text', '') or ""
#             entities = row.get('entities_text', '') or ""
#             doc_str = f"- Title: {title}\n  Intent: {intent}\n  Entities: {entities}\n"
#             docs.append(doc_str)
#         joined_docs = "\n".join(docs)

#         return f"""
#     You are analyzing a microcluster of documents. Each document contains several fields
#     (title, search intent, entities, etc.).

#     Cluster ID: {cluster_id}, Microcluster ID: {micro_id}

#     Here are the documents from this microcluster:
#     ---
#     {joined_docs}
#     ---

#     Please create a structured JSON object with the following fields:

#     {{
#     "summary": "A 2–3 sentence description of the main theme of this microcluster.",
#     "key_points": ["3–6 bullet points highlighting key recurring ideas or topics."],
#     "keywords": ["10–20 representative keywords or phrases (short, lowercase)."],
#     "microcluster_name": "A short 2–5 word descriptive label for this microcluster."
#     }}

#     Rules:
#     - Do not invent facts not present in the documents.
#     - Use clear, concise English.
#     - Keywords should reflect actual terms from the text.
#     - The microcluster name should be broad enough to describe the set, but specific enough to distinguish it.
#     """.strip()

#     def summarize_microcluster(df_subset: pd.DataFrame, cluster_id: int, micro_id: Any,
#                            batch_size: int = 25) -> ast.Dict[str, Any]:
#         result = {"mode": None, "prompts": [], "info": {}}
#         n = len(df_subset)
#         if n == 0:
#             result["mode"] = "empty"
#             result["info"] = {"n_docs": 0}
#             return result

#         df_local = df_subset.copy().reset_index(drop=False)
#         df_local["__emb"] = df_local["combined_embedding"].apply(lambda x: np.asarray(x, dtype=float))
#         result["info"]["n_docs"] = n

#         if n < 10:
#             result["mode"] = "small"
#             result["prompts"].append({
#                 "prompt_id": f"{micro_id}_all",
#                 "prompt": build_micro_prompt(df_local, cluster_id, micro_id),
#                 "doc_indices": df_local["index"].tolist(),
#                 "doc_ids": df_local["id"].tolist()
#             })
#             result["info"]["selection"] = "all_docs"

#         elif n <= 30:
#             result["mode"] = "medium"
#             emb_stack = np.vstack(df_local["__emb"].values)
#             centroid = emb_stack.mean(axis=0)
#             df_local["__dist"] = df_local["__emb"].apply(lambda v: np.linalg.norm(v - centroid))
#             top_df = df_local.nsmallest(10, "__dist")
#             result["prompts"].append({
#                 "prompt_id": f"{micro_id}_top10",
#                 "prompt": build_micro_prompt(top_df, cluster_id, micro_id),
#                 "doc_indices": top_df["index"].tolist(),
#                 "doc_ids": top_df["id"].tolist()
#             })
#             result["info"]["selection"] = "topk_centroid"

#         else:
#             result["mode"] = "large"
#             batches = np.array_split(df_local, math.ceil(n / batch_size))
#             for idx, batch in enumerate(batches, start=1):
#                 result["prompts"].append({
#                     "prompt_id": f"{micro_id}_part{idx}",
#                     "prompt": build_micro_prompt(batch, cluster_id, f"{micro_id}_part{idx}"),
#                     "doc_indices": batch["index"].tolist(),
#                     "doc_ids": batch["id"].tolist()
#                 })
#             result["info"]["selection"] = "batches"

#         return result
    
#     def save_microcluster_result(cluster_id, microcluster_id, llm_result: dict):
#         summary = llm_result.get("summary", "")
#         key_points = llm_result.get("key_points", [])
#         keywords = llm_result.get("keywords", [])
#         micro_name = llm_result.get("microcluster_name", "")

#         if isinstance(key_points, list):
#             key_points = "; ".join([str(k) for k in key_points])
#         if isinstance(keywords, list):
#             keywords = ", ".join([str(k) for k in keywords])

#         representative_text = summary
#         if key_points:
#             representative_text += "\n\n- " + "\n- ".join(key_points.split("; "))

#         data = {
#             "cluster_id": cluster_id,
#             "microcluster_id": microcluster_id,
#             "microcluster_name": micro_name,
#             "representative_text": representative_text,
#             "representative_keywords": keywords,
#         }

#         result = supabase.table("cluster_micro") \
#                         .update(data) \
#                         .eq("cluster_id", cluster_id) \
#                         .eq("microcluster_id", microcluster_id) \
#                         .execute()
#         print("✅ Saved to Supabase:", result)

#     # fetch data for the target cluster/microcluster
#     base_sql = """
#     SELECT ca.doc_id AS id,
#            d.suggested_title,
#            d.user_search_intent,
#            d.entities,
#            v.suggested_title AS suggested_title_embedding,
#            v.metadata AS metadata_embedding,
#            v.user_search_intent AS user_search_intent_embedding,
#            v.faq_pairs AS faq_pairs_embedding,
#            v.entities AS entities_embedding
#     FROM public.cluster_assignments ca
#     JOIN public.data_text d ON d.id = ca.doc_id
#     JOIN vecs.data_vector v ON v.id = ca.doc_id
#     WHERE ca.cluster_id = %s
#     """

#     params = [request.target_cluster_id]
#     if request.target_microcluster_id:
#         base_sql += " AND ca.microcluster_id = %s"
#         params.append(request.target_microcluster_id)

#     df = pd.read_sql(base_sql, cur, params=params)
#     for col in [
#         "suggested_title_embedding",
#         "metadata_embedding",
#         "user_search_intent_embedding",
#         "faq_pairs_embedding",
#         "entities_embedding",
#     ]:
#         df[col] = df[col].apply(parse_embedding)

#     df["combined_embedding"] = df.apply(
#         lambda r: np.mean(
#             [
#                 r["suggested_title_embedding"],
#                 r["metadata_embedding"],
#                 r["user_search_intent_embedding"],
#                 r["faq_pairs_embedding"],
#                 r["entities_embedding"],
#             ],
#             axis=0,
#         ),
#         axis=1,
#     )
#     df["entities_text"] = df["entities_text"].apply(lambda x: str(x) if x else "")

#     # build prompt
#     prompt = build_micro_prompt(df, request.target_cluster_id, request.target_microcluster_id)

#     # call LLM
#     llm_result = run_llm(prompt)

#     # --- Save result to supabase ---
#     cur.table("cluster_micro").upsert({
#         "cluster_id": request.target_cluster_id,
#         "microcluster_id": request.target_microcluster_id,
#         "microcluster_name": llm_result.get("microcluster_name", ""),
#         "representative_text": llm_result.get("summary", ""),
#         "representative_keywords": ", ".join(llm_result.get("keywords", [])),
#     }).execute()

#     return {"status": "success", "cluster_id": request.target_cluster_id, "micro_id": request.target_microcluster_id, "llm_result": llm_result}


# # ========== Stage 3 Endpoint : Macrocluster ==========
# class MacroclusterRequest(BaseModel):
#     cluster_id: int
#     batch_size: int = 25  # optional, default 25

# @app.post("/summarize_macrocluster")
# async def summarize_macrocluster_endpoint(request: MacroclusterRequest):

#     def get_microclusters_for_cluster(cluster_id: int):
#         response = supabase.table("cluster_micro").select(
#             "microcluster_id, microcluster_name, representative_text, representative_keywords"
#         ).eq("cluster_id", cluster_id).execute()

#         if not response.data:
#             print(f"⚠️ No microclusters found for cluster_id {cluster_id}")
#             return []

#         print(f"✅ Retrieved {len(response.data)} microclusters for cluster {cluster_id}")
#         return response.data

#     def build_macro_prompt(cluster_id: int, microclusters: list, batch_id=None) -> str:
#         parts = []
#         for m in microclusters:
#             name = m.get("microcluster_name", "")
#             text = m.get("representative_text", "")
#             keywords = m.get("representative_keywords", "")
#             parts.append(
#                 f"- Microcluster {m['microcluster_id']} ({name}):\n"
#                 f"  Summary: {text}\n"
#                 f"  Keywords: {keywords}\n"
#             )

#         joined = "\n\n".join(parts)

#         prompt = f"""
#     You are analyzing a macrocluster of documents.
#     This macrocluster contains several microclusters, each with a summary and keywords.

#     Cluster ID: {cluster_id}{f", Batch {batch_id}" if batch_id else ""}

#     Here are the microclusters:
#     ---
#     {joined}
#     ---

#     Please create a structured JSON object with the following fields:

#     {{
#     "summary": "A 2–3 sentence summary of the main theme of this batch of microclusters.",
#     "key_points": ["3–6 bullet points highlighting recurring themes."],
#     "keywords": ["10–20 representative keywords or phrases (short, lowercase)."],
#     "batch_name": "A short 2–5 word descriptive label for this batch."
#     }}

#     Rules:
#     - Do not invent facts not present in the microclusters.
#     - Use clear, concise English.
#     - Keywords should be actual recurring terms across microclusters.
#     - The batch_name should be broad but specific enough to describe the set.
#     """.strip()

#         return prompt


#     def build_reduce_prompt(cluster_id: int, batch_results: list) -> str:
#         parts = []
#         for i, br in enumerate(batch_results, start=1):
#             summary = br["llm_result"].get("summary", "")
#             key_points = br["llm_result"].get("key_points", [])
#             keywords = br["llm_result"].get("keywords", [])
#             batch_name = br["llm_result"].get("batch_name", "")
#             parts.append(
#                 f"- Batch {i} ({batch_name}):\n"
#                 f"  Summary: {summary}\n"
#                 f"  Key Points: {key_points}\n"
#                 f"  Keywords: {keywords}\n"
#             )

#         joined = "\n\n".join(parts)

#         prompt = f"""
#     You are analyzing a macrocluster of documents.

#     Cluster ID: {cluster_id}

#     You are given summaries of several batches of microclusters.
#     Each batch has its own summary, key points, and keywords.

#     Here are the batch-level summaries:
#     ---
#     {joined}
#     ---

#     Please create a structured JSON object with the following fields:

#     {{
#     "summary": "A 3–5 sentence integrated summary of the entire macrocluster.",
#     "key_points": ["5–8 key points synthesizing the recurring ideas across batches."],
#     "keywords": ["15–25 representative keywords or phrases (short, lowercase)."],
#     "cluster_name": "A short 2–5 word descriptive label for the entire macrocluster."
#     }}

#     Rules:
#     - Focus on themes that recur across multiple batches.
#     - Do not repeat batch-specific details unless central to the overall cluster.
#     - Keywords should represent the macrocluster as a whole.
#     - The cluster_name should be broad but distinctive.
#     """.strip()

#         return prompt
    
#     def save_macrocluster_result(cluster_id: int, final_result: dict):
#         summary = final_result.get("summary", "")
#         key_points = final_result.get("key_points", [])
#         keywords = final_result.get("keywords", [])

#         # Handle cluster_name correctly
#         cluster_name = final_result.get("cluster_name")
#         if not cluster_name:
#             cluster_name = final_result.get("batch_name", f"Cluster {cluster_id}")  # ✅ fallback

#         # Representative text
#         representative_text = summary
#         if isinstance(key_points, list) and key_points:
#             representative_text += "\n\n- " + "\n- ".join(key_points)

#         # Keywords format
#         if isinstance(keywords, list):
#             keywords_str = ", ".join(keywords)
#         else:
#             keywords_str = str(keywords)

#         data = {
#             "cluster_id": cluster_id,
#             "cluster_name": cluster_name,  # always filled now ✅
#             "representative_text": representative_text,
#             "representative_keywords": keywords_str,
#         }

#         result = supabase.table("cluster_macro").upsert(data).execute()
#         print("✅ Saved to Supabase:", result)

#     def summarize_macrocluster(cluster_id: int, batch_size: int = 25):
#         microclusters = get_microclusters_for_cluster(cluster_id)
#         if not microclusters:
#             return {}

#         # Stage 1: split into batches
#         n = len(microclusters)
#         n_batches = math.ceil(n / batch_size)
#         batch_results = []

#         for i in range(n_batches):
#             start = i * batch_size
#             end = min((i + 1) * batch_size, n)
#             batch_microclusters = microclusters[start:end]

#             prompt_text = build_macro_prompt(cluster_id, batch_microclusters, batch_id=i+1)
#             print("\n" + "=" * 60)
#             print(f"⚡ Running LLM for {cluster_id}_batch{i+1} ({len(batch_microclusters)} microclusters)")
#             llm_result = run_llm(prompt_text)
#             batch_results.append({"prompt_id": f"{cluster_id}_batch{i+1}", "llm_result": llm_result})

#         # Stage 2: reduce if multiple batches
#         if len(batch_results) > 1:
#             reduce_prompt = build_reduce_prompt(cluster_id, batch_results)
#             print("\n⚡ Running LLM for FINAL REDUCE step")
#             final_result = run_llm(reduce_prompt)
#         else:
#             final_result = batch_results[0]["llm_result"]

#         # Save into Supabase
#         save_macrocluster_result(cluster_id, final_result)

#         return {
#             "cluster_id": cluster_id,
#             "batch_results": batch_results,
#             "final_result": final_result
#         }

#     try:
#         result = summarize_macrocluster(
#             cluster_id=request.cluster_id,
#             batch_size=request.batch_size
#         )

#         if not result:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"No microclusters found for cluster_id {request.cluster_id}"
#             )

#         return {
#             "status": "success",
#             "data": result
#         }

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error summarizing macrocluster: {str(e)}"
#         )


    
    


