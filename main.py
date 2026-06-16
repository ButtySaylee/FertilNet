import os
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    data_path = os.path.join(BASE_DIR, "data", "soil_data.csv")
    batch_size = 32
    teacher_epochs = 50
    assistant_epochs = 35
    student_epochs = 40
    lr_teacher = 1e-3
    lr_assistant = 5e-4
    lr_student = 1e-4
    alpha = 0.7
    beta_assistant = 0.2
    temperature = 5.0
    rf_trees = 300
    seed = 42
    latency_runs = 300
    warmup_runs = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outputs_dir = os.path.join(BASE_DIR, "outputs")

cfg = Config()
os.makedirs(cfg.outputs_dir, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# DATASET LOADING
# -------------------------
def load_dataset():
    df = pd.read_csv(cfg.data_path)

    feature_cols = ["N","P","K","pH","EC","OC","S","Zn","Fe","Cu","Mn","B"]
    label_col = "Output"

    X = df[feature_cols].values
    y = df[label_col].values

    le = LabelEncoder()
    y = le.fit_transform(y)

    # Keep an exact 80/10/10 split to align with report claims.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=cfg.seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=1 / 9, stratify=y_train_full, random_state=cfg.seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, le, df

class SoilDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# -------------------------
# MODEL DEFINITIONS
# -------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, layers=3, dropout=0.3):
        super().__init__()
        modules = []
        dim = input_dim
        for _ in range(layers):
            modules.append(nn.Linear(dim, hidden_dim))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            dim = hidden_dim
        modules.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

# -------------------------
# EVALUATION
# -------------------------
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            logits = model(xb)
            pred = logits.argmax(1)
            preds.extend(pred.cpu().numpy())
            targets.extend(yb.cpu().numpy())
    return accuracy_score(targets, preds)


def predict_model(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            logits = model(xb)
            pred = logits.argmax(1)
            preds.extend(pred.cpu().numpy())
            targets.extend(yb.cpu().numpy())
    return np.array(targets), np.array(preds)


def metrics_dict(y_true, y_pred):
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred, labels, title, out_file):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=labels, yticklabels=labels,
           ylabel="True label", xlabel="Predicted label", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close(fig)
    return cm.tolist()


def model_param_count(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def file_size_mb(path):
    if not os.path.exists(path):
        return None
    return round(os.path.getsize(path) / (1024 * 1024), 6)


def benchmark_pytorch_latency(model, input_dim):
    model.eval()
    x = torch.randn(1, input_dim).to(cfg.device)
    with torch.no_grad():
        for _ in range(cfg.warmup_runs):
            _ = model(x)

    times = []
    with torch.no_grad():
        for _ in range(cfg.latency_runs):
            t0 = time.perf_counter()
            _ = model(x)
            if cfg.device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def benchmark_onnx_latency(onnx_path, input_dim):
    if ort is None or not os.path.exists(onnx_path):
        return None
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    x = np.random.randn(1, input_dim).astype(np.float32)

    for _ in range(cfg.warmup_runs):
        _ = session.run(None, {inp_name: x})

    times = []
    for _ in range(cfg.latency_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {inp_name: x})
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def train_baselines(X_train, y_train, X_test, y_test):
    results = {}

    rf = RandomForestClassifier(n_estimators=cfg.rf_trees, random_state=cfg.seed)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["random_forest"] = metrics_dict(y_test, rf_pred)

    if XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            num_class=len(np.unique(y_train)),
            random_state=cfg.seed,
            eval_metric="mlogloss",
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        results["xgboost"] = metrics_dict(y_test, xgb_pred)
    else:
        results["xgboost"] = {"note": "xgboost package not available"}

    return results

# -------------------------
# TEACHER TRAINING
# -------------------------
def train_teacher(model, train_loader, val_loader, best_path=None, tag="Teacher"):
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_teacher)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0
    if best_path is None:
        best_path = f"{cfg.outputs_dir}/teacher_best.pth"

    for epoch in range(cfg.teacher_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_acc = evaluate(model, val_loader)
        print(f"[{tag}] Epoch {epoch+1}  Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path))
    return model


def train_assistant(assistant, teacher, train_loader, val_loader):
    assistant.to(cfg.device)
    teacher.to(cfg.device)
    teacher.eval()

    opt = torch.optim.Adam(assistant.parameters(), lr=cfg.lr_assistant)
    ce_loss = nn.CrossEntropyLoss()
    T = cfg.temperature
    alpha = cfg.alpha

    best_acc = 0
    best_path = f"{cfg.outputs_dir}/assistant_best.pth"

    for epoch in range(cfg.assistant_epochs):
        assistant.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            assistant_logits = assistant(xb)
            with torch.no_grad():
                teacher_logits = teacher(xb)

            loss_hard = ce_loss(assistant_logits, yb)
            assistant_log_probs = F.log_softmax(assistant_logits / T, dim=1)
            teacher_probs = F.softmax(teacher_logits / T, dim=1)
            loss_soft = F.kl_div(assistant_log_probs, teacher_probs, reduction='batchmean') * (T * T)
            loss = alpha * loss_hard + (1 - alpha) * loss_soft

            opt.zero_grad()
            loss.backward()
            opt.step()

        val_acc = evaluate(assistant, val_loader)
        print(f"[Assistant] Epoch {epoch+1}  Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(assistant.state_dict(), best_path)

    assistant.load_state_dict(torch.load(best_path))
    return assistant

# -------------------------
# STUDENT TRAINING (KD)
# -------------------------
def train_student(student, teacher, assistant, train_loader, val_loader):
    student.to(cfg.device)
    teacher.to(cfg.device)
    teacher.eval()
    assistant.eval()

    opt = torch.optim.Adam(student.parameters(), lr=cfg.lr_student)
    ce_loss = nn.CrossEntropyLoss()
    T = cfg.temperature
    alpha = cfg.alpha
    beta = cfg.beta_assistant

    best_acc = 0
    best_path = f"{cfg.outputs_dir}/student_best.pth"

    for epoch in range(cfg.student_epochs):
        student.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)

            student_logits = student(xb)

            with torch.no_grad():
                teacher_logits = teacher(xb)
                assistant_logits = assistant(xb)

            loss_hard = ce_loss(student_logits, yb)

            student_log_probs = F.log_softmax(student_logits / T, dim=1)
            teacher_probs = F.softmax(teacher_logits / T, dim=1)
            loss_soft = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T*T)
            assistant_probs = F.softmax(assistant_logits / T, dim=1)
            loss_soft_assistant = F.kl_div(student_log_probs, assistant_probs, reduction='batchmean') * (T*T)

            loss = alpha * loss_hard + beta * loss_soft_assistant + (1 - alpha - beta) * loss_soft

            opt.zero_grad()
            loss.backward()
            opt.step()

        val_acc = evaluate(student, val_loader)
        print(f"[Student] Epoch {epoch+1}  Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), best_path)

    student.load_state_dict(torch.load(best_path))
    return student

# -------------------------
# ONNX EXPORT
# -------------------------
def export_onnx(model, input_dim):
    model.eval()
    dummy = torch.randn(1, input_dim).to(cfg.device)
    path = f"{cfg.outputs_dir}/student.onnx"
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["output"])
    print("Exported ONNX model to:", path)
    return path

# -------------------------
# MAIN PIPELINE
# -------------------------
def main():
    set_seed(cfg.seed)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, le, df = load_dataset()

    train_loader = DataLoader(SoilDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(SoilDataset(X_val, y_val), batch_size=cfg.batch_size)
    test_loader = DataLoader(SoilDataset(X_test, y_test), batch_size=cfg.batch_size)

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    teacher = MLP(input_dim, hidden_dim=256, num_classes=num_classes, layers=5)
    assistant = MLP(input_dim, hidden_dim=128, num_classes=num_classes, layers=3)
    student = MLP(input_dim, hidden_dim=64, num_classes=num_classes, layers=2)
    student_no_kd = MLP(input_dim, hidden_dim=64, num_classes=num_classes, layers=2)

    print("\nTraining Teacher...")
    teacher = train_teacher(teacher, train_loader, val_loader)

    print("\nTraining Assistant with KD...")
    assistant = train_assistant(assistant, teacher, train_loader, val_loader)

    print("\nTraining Student (no KD baseline)...")
    student_no_kd = train_teacher(
        student_no_kd,
        train_loader,
        val_loader,
        best_path=f"{cfg.outputs_dir}/student_nokd_best.pth",
        tag="Student_NoKD",
    )

    print("\nTraining Student with Hierarchical KD...")
    student = train_student(student, teacher, assistant, train_loader, val_loader)

    print("\nFinal Evaluation:")
    y_t, y_p_teacher = predict_model(teacher, test_loader)
    _, y_p_assistant = predict_model(assistant, test_loader)
    _, y_p_student = predict_model(student, test_loader)
    _, y_p_student_nokd = predict_model(student_no_kd, test_loader)

    teacher_metrics = metrics_dict(y_t, y_p_teacher)
    assistant_metrics = metrics_dict(y_t, y_p_assistant)
    student_metrics = metrics_dict(y_t, y_p_student)
    student_nokd_metrics = metrics_dict(y_t, y_p_student_nokd)

    print("Teacher Test Accuracy:", teacher_metrics["accuracy"])
    print("Assistant Test Accuracy:", assistant_metrics["accuracy"])
    print("Student (HierKD) Test Accuracy:", student_metrics["accuracy"])
    print("Student (No KD) Test Accuracy:", student_nokd_metrics["accuracy"])

    baseline_results = train_baselines(X_train, y_train, X_test, y_test)
    print("Random Forest Accuracy:", baseline_results["random_forest"].get("accuracy"))

    cm_teacher = save_confusion_matrix(y_t, y_p_teacher, list(le.classes_), "Teacher Confusion Matrix", f"{cfg.outputs_dir}/cm_teacher.png")
    cm_assistant = save_confusion_matrix(y_t, y_p_assistant, list(le.classes_), "Assistant Confusion Matrix", f"{cfg.outputs_dir}/cm_assistant.png")
    cm_student = save_confusion_matrix(y_t, y_p_student, list(le.classes_), "Student HierKD Confusion Matrix", f"{cfg.outputs_dir}/cm_student_hierkd.png")
    cm_student_nokd = save_confusion_matrix(y_t, y_p_student_nokd, list(le.classes_), "Student No-KD Confusion Matrix", f"{cfg.outputs_dir}/cm_student_nokd.png")

    onnx_path = export_onnx(student, input_dim)

    pytorch_latency = benchmark_pytorch_latency(student, input_dim)
    onnx_latency = benchmark_onnx_latency(onnx_path, input_dim)

    results = {
        "config": {
            "seed": cfg.seed,
            "alpha": cfg.alpha,
            "beta_assistant": cfg.beta_assistant,
            "temperature": cfg.temperature,
            "split": "80/10/10 stratified",
        },
        "dataset": {
            "rows": int(df.shape[0]),
            "features": int(X_train.shape[1]),
            "classes": [str(c) for c in le.classes_],
        },
        "models": {
            "teacher": {
                "params": model_param_count(teacher),
                "pth_size_mb": file_size_mb(f"{cfg.outputs_dir}/teacher_best.pth"),
                **teacher_metrics,
                "confusion_matrix": cm_teacher,
            },
            "assistant": {
                "params": model_param_count(assistant),
                "pth_size_mb": file_size_mb(f"{cfg.outputs_dir}/assistant_best.pth"),
                **assistant_metrics,
                "confusion_matrix": cm_assistant,
            },
            "student_hierkd": {
                "params": model_param_count(student),
                "pth_size_mb": file_size_mb(f"{cfg.outputs_dir}/student_best.pth"),
                "onnx_size_mb": file_size_mb(onnx_path),
                **student_metrics,
                "confusion_matrix": cm_student,
                "latency_ms_pytorch": pytorch_latency,
                "latency_ms_onnxruntime": onnx_latency,
            },
            "student_no_kd": {
                "params": model_param_count(student_no_kd),
                "pth_size_mb": file_size_mb(f"{cfg.outputs_dir}/student_nokd_best.pth"),
                **student_nokd_metrics,
                "confusion_matrix": cm_student_nokd,
            },
        },
        "baselines": baseline_results,
    }

    with open(f"{cfg.outputs_dir}/metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    table_rows = [
        {
            "model": "Teacher",
            "accuracy": teacher_metrics["accuracy"],
            "precision_macro": teacher_metrics["precision_macro"],
            "recall_macro": teacher_metrics["recall_macro"],
            "f1_macro": teacher_metrics["f1_macro"],
            "params": model_param_count(teacher),
            "size_mb": file_size_mb(f"{cfg.outputs_dir}/teacher_best.pth"),
        },
        {
            "model": "Assistant",
            "accuracy": assistant_metrics["accuracy"],
            "precision_macro": assistant_metrics["precision_macro"],
            "recall_macro": assistant_metrics["recall_macro"],
            "f1_macro": assistant_metrics["f1_macro"],
            "params": model_param_count(assistant),
            "size_mb": file_size_mb(f"{cfg.outputs_dir}/assistant_best.pth"),
        },
        {
            "model": "Student_HierKD",
            "accuracy": student_metrics["accuracy"],
            "precision_macro": student_metrics["precision_macro"],
            "recall_macro": student_metrics["recall_macro"],
            "f1_macro": student_metrics["f1_macro"],
            "params": model_param_count(student),
            "size_mb": file_size_mb(f"{cfg.outputs_dir}/student_best.pth"),
            "onnx_size_mb": file_size_mb(onnx_path),
            "latency_pytorch_mean_ms": pytorch_latency["mean_ms"],
            "latency_onnx_mean_ms": None if onnx_latency is None else onnx_latency["mean_ms"],
        },
        {
            "model": "Student_NoKD",
            "accuracy": student_nokd_metrics["accuracy"],
            "precision_macro": student_nokd_metrics["precision_macro"],
            "recall_macro": student_nokd_metrics["recall_macro"],
            "f1_macro": student_nokd_metrics["f1_macro"],
            "params": model_param_count(student_no_kd),
            "size_mb": file_size_mb(f"{cfg.outputs_dir}/student_nokd_best.pth"),
        },
        {
            "model": "RandomForest",
            "accuracy": baseline_results["random_forest"].get("accuracy"),
            "precision_macro": baseline_results["random_forest"].get("precision_macro"),
            "recall_macro": baseline_results["random_forest"].get("recall_macro"),
            "f1_macro": baseline_results["random_forest"].get("f1_macro"),
        },
    ]

    if isinstance(baseline_results.get("xgboost"), dict) and "accuracy" in baseline_results["xgboost"]:
        table_rows.append(
            {
                "model": "XGBoost",
                "accuracy": baseline_results["xgboost"].get("accuracy"),
                "precision_macro": baseline_results["xgboost"].get("precision_macro"),
                "recall_macro": baseline_results["xgboost"].get("recall_macro"),
                "f1_macro": baseline_results["xgboost"].get("f1_macro"),
            }
        )

    pd.DataFrame(table_rows).to_csv(f"{cfg.outputs_dir}/results_table.csv", index=False)
    print(f"Saved publication-ready metrics to {cfg.outputs_dir}/metrics_summary.json")
    print(f"Saved publication-ready table to {cfg.outputs_dir}/results_table.csv")

if __name__ == "__main__":
    main()