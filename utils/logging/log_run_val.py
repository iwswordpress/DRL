# %%
# https://pynative.com/python-sqlite-insert-into-table/
import sqlite3
import uuid
"""Given a dictionary of values, inserts run into tbl runs."""


def log_run_val(vals):

    uuid_val = str(uuid.uuid4())
    model = vals.get('model', 'LogReg')
    dataset = vals.get('dataset', 'mlr.csv')
    features = vals.get('features', 'all')
    accuracy = vals.get('accuracy', 0.0)
    precision = vals.get('precision', 0.0)
    recall = vals.get('recall', 0.0)
    roc_auc = vals.get('roc_auc', 0.0)
    f1 = vals.get('f1', 0.0)
    params = vals.get('params', '{}')
    notes = vals.get('notes', 'EMPTY')
    db = vals.get('db', '../db/ml.db')
    project = vals.get('project', 'NIH-GER')
    analyst = vals.get('analyst', 'Craig West')

    sql = f"INSERT INTO runs (ID,model,dataset,features,accuracy, precision, recall, roc_auc, f1, params, notes,project, analyst) VALUES('{uuid_val}','{model}','{dataset}','{features}',{accuracy},{precision},{recall},{roc_auc},{f1},'{params}','{notes}','{project}','{analyst}')"

    print(sql)

    try:
        sqliteConnection = sqlite3.connect(db)
        cursor = sqliteConnection.cursor()
        print("Successfully Connected to SQLite")

        cursor.execute(sql)
        sqliteConnection.commit()
        print(">>> Record inserted successfully into runs table ",
              cursor.rowcount)
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert data into sqlite table", error)
    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("The SQLite connection is closed")

# %%

#  TEST


vals = {
    'model': '2MONDAY LOG_RUN_VAL',
    'dataset': 'GER-HIGH',
    'features': 'top 1000',
    'accuracy': 0.2,
    'precision': 0.4,
    'recall': 0.6,
    'roc_auc': 0.6,
    'f1': 0.6,
    'params':  '{"max_iter": 100}',
    'notes': 'some notes',
    'db': '../db/ml.db',
    'project': 'NIH-USA',
    'analyst': 'Craig West'
}

# log_run_val(vals)

# %%
