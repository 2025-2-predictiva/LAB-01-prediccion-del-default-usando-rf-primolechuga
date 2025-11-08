# flake8: noqa: E501
#

import os
import gzip
import json
import pickle
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix


def load_data(file_path: str) -> pd.DataFrame:
	return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
	df = df.rename(columns={"default payment next month": "default"})
	if "ID" in df.columns:
		df = df.drop(columns=["ID"])
	df = df.dropna()
	if "EDUCATION" in df.columns:
		df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
	return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
	binary_cols = [
		c for c in X.columns if set(X[c].dropna().unique()).issubset({0, 1})
	]
	cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
	cat_cols = list(dict.fromkeys(cat_cols + binary_cols))
	num_cols = [c for c in X.columns if c not in cat_cols]

	numeric_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler())
	])

	categorical_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
		("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse=False))
	])

	preprocessor = ColumnTransformer(transformers=[
		("num", numeric_transformer, num_cols),
		("cat", categorical_transformer, cat_cols)
	], remainder="drop")

	pipeline = Pipeline(steps=[
		("preproc", preprocessor),
		("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
	])

	return pipeline


def optimize_and_save(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_test: pd.DataFrame,
	y_test: pd.Series,
	model_path: str = "files/models/model.pkl.gz",
	metrics_path: str = "files/output/metrics.json",
) -> GridSearchCV:
	pipeline = build_pipeline(X_train)

	param_grid = {
		"classifier__n_estimators": [200, 300],
		"classifier__max_depth": [18, 20],
		"classifier__min_samples_split": [3, 5],
		"classifier__min_samples_leaf": [1, 3],
		"classifier__class_weight": [
			{0: 1, 1: 1.0},
			{0: 1, 1: 1.5},
			{0: 1, 1: 2.5},
			{0: 1, 1: 2.9},
			{0: 1, 1: 3.0},
			{0: 1, 1: 3.1},
			{0: 1, 1: 3.15},
		],
	}

	cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

	grid = GridSearchCV(
		estimator=pipeline,
		param_grid=param_grid,
		scoring="balanced_accuracy",
		cv=cv,
		n_jobs=-1,
		verbose=1,
		refit=True,
	)

	grid.fit(X_train, y_train)

	os.makedirs(os.path.dirname(model_path), exist_ok=True)
	os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

	with gzip.open(model_path, "wb") as f:
		pickle.dump(grid, f)

	rows = []
	for ds_name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
		y_pred = grid.predict(X)
		prec = float(precision_score(y, y_pred, zero_division=0))
		bal_acc = float(balanced_accuracy_score(y, y_pred))
		rec = float(recall_score(y, y_pred, zero_division=0))
		f1 = float(f1_score(y, y_pred, zero_division=0))
		rows.append({
			"type": "metrics",
			"dataset": ds_name,
			"precision": prec,
			"balanced_accuracy": bal_acc,
			"recall": rec,
			"f1_score": f1,
		})

	cm_train = confusion_matrix(y_train, grid.predict(X_train))
	cm_test = confusion_matrix(y_test, grid.predict(X_test))

	rows.append({
		"type": "cm_matrix",
		"dataset": "train",
		"true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
		"true_1": {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])},
	})
	rows.append({
		"type": "cm_matrix",
		"dataset": "test",
		"true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
		"true_1": {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])},
	})

	with open(metrics_path, "w", encoding="utf-8") as f:
		for r in rows:
			f.write(json.dumps(r) + "\n")

	return grid


if __name__ == "__main__":
	# Small runner: use files/input CSVs if present
	traindf_path = "files/input/train_default_of_credit_card_clients.csv"
	testdf_path = "files/input/test_default_of_credit_card_clients.csv"
	if os.path.exists(traindf_path) and os.path.exists(testdf_path):
		train_df = load_data(traindf_path)
		test_df = load_data(testdf_path)
		train_df = clean_data(train_df)
		test_df = clean_data(test_df)

		X_train = train_df.drop(columns=["default"])
		y_train = train_df["default"]
		X_test = test_df.drop(columns=["default"])
		y_test = test_df["default"]

		optimize_and_save(X_train, y_train, X_test, y_test)

