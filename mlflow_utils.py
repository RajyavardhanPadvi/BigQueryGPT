# mlflow_utils.py
import mlflow, os, json, joblib
def init_mlflow(tracking_uri: str = None, experiment_name: str = "/Shared/dataspark"):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def log_index_run(input_file: str, docs: list, vstore, run_name: str = "dataspark_index") -> str:
    tmp_dir = "/tmp/dataspark_artifacts"
    os.makedirs(tmp_dir, exist_ok=True)
    joblib.dump(vstore.vectorizer, os.path.join(tmp_dir, "tfidf.joblib"))
    with open(os.path.join(tmp_dir, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(getattr(vstore, 'docs', docs), f, ensure_ascii=False)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("input_file", input_file)
        mlflow.log_metric("n_docs", len(docs))
        mlflow.log_artifact(os.path.join(tmp_dir, "tfidf.joblib"), artifact_path="artifacts")
        mlflow.log_artifact(os.path.join(tmp_dir, "docs.json"), artifact_path="artifacts")
        return run.info.run_id
