import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "",  # Change this if your MySQL root account has a password.
}
DATABASE_NAME = "banking_fraud_db"


def get_server_connection():
    return mysql.connector.connect(**DB_CONFIG)


def get_connection():
    config = {**DB_CONFIG, "database": DATABASE_NAME}
    return mysql.connector.connect(**config)


def initialize_database():
    """Create the database and predictions table if they do not exist."""
    connection = None
    cursor = None
    try:
        connection = get_server_connection()
        cursor = connection.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{DATABASE_NAME}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        connection.commit()
    except Error as exc:
        raise RuntimeError(f"Could not create/connect to MySQL database: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                transaction_time DOUBLE NOT NULL,
                amount DOUBLE NOT NULL,
                actual_class TINYINT NULL,
                predicted_class TINYINT NOT NULL,
                fraud_probability DOUBLE NOT NULL,
                input_type VARCHAR(40) NOT NULL DEFAULT 'manual_demo',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Safe migration for a table created by an earlier version.
        cursor.execute("SHOW COLUMNS FROM transactions LIKE 'input_type'")
        if cursor.fetchone() is None:
            cursor.execute(
                "ALTER TABLE transactions ADD COLUMN input_type "
                "VARCHAR(40) NOT NULL DEFAULT 'manual_demo'"
            )
        connection.commit()
    except Error as exc:
        raise RuntimeError(f"Could not initialize transactions table: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


def save_transaction(
    transaction_time,
    amount,
    actual_class,
    predicted_class,
    fraud_probability,
    input_type="manual_demo",
):
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO transactions
            (transaction_time, amount, actual_class, predicted_class,
             fraud_probability, input_type)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                float(transaction_time),
                float(amount),
                None if actual_class is None else int(actual_class),
                int(predicted_class),
                float(fraud_probability),
                str(input_type),
            ),
        )
        connection.commit()
        return cursor.lastrowid
    except Error as exc:
        if connection:
            connection.rollback()
        raise RuntimeError(f"Could not save prediction: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


def get_transactions(limit=500):
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT id, transaction_time, amount, actual_class, predicted_class,
                   fraud_probability, input_type, created_at
            FROM transactions
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (int(limit),),
        )
        return cursor.fetchall()
    except Error as exc:
        raise RuntimeError(f"Could not load prediction history: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


def get_statistics():
    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT
                COUNT(*) AS saved_predictions,
                COALESCE(SUM(CASE WHEN predicted_class = 1 THEN 1 ELSE 0 END), 0)
                    AS predicted_fraud,
                COALESCE(SUM(CASE WHEN predicted_class = 0 THEN 1 ELSE 0 END), 0)
                    AS predicted_legitimate,
                COALESCE(AVG(fraud_probability), 0) AS average_fraud_probability
            FROM transactions
            """
        )
        row = cursor.fetchone()
        return row or {
            "saved_predictions": 0,
            "predicted_fraud": 0,
            "predicted_legitimate": 0,
            "average_fraud_probability": 0,
        }
    except Error as exc:
        raise RuntimeError(f"Could not load dashboard statistics: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()
