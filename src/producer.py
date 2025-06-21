import pika
import json
import csv
import time
import os

# RabbitMQ Configuration
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST', 'localhost')
RABBITMQ_QUEUE = 'loan_application_queue'

# Data source for new applications
NEW_APPLICATIONS_CSV = './data/new_applications.csv' 

def send_message_to_queue(channel, message_body):
    """Sends a single message to the specified queue."""
    try:
        channel.basic_publish(
            exchange='',
            routing_key=RABBITMQ_QUEUE,
            body=json.dumps(message_body),
            properties=pika.BasicProperties(
                delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE  # Make message persistent
            )
        )
        print(f" [x] Sent application: {message_body.get('customer_id', 'Unknown ID')}")
    except Exception as e:
        print(f"Error sending message: {e}")

def main():
    # Create a dummy new_applications.csv if it doesn't exist
    if not os.path.exists(NEW_APPLICATIONS_CSV):
        print(f"'{NEW_APPLICATIONS_CSV}' not found. Creating a dummy file for demonstration.")
        os.makedirs(os.path.dirname(NEW_APPLICATIONS_CSV), exist_ok=True)
        header = [
            "customer_id", "emp_title", "state", "annual_inc", "annual_inc_joint", "type", "home_ownership", 
            "loan_amount", "term", "int_rate", "grade", "issue_d", "purpose", "installment", "avg_cur_bal", "tot_cur_bal", "job_level",
        ]
        # Example data row 
        dummy_data_row = [
            "C2", "Engineer", "NY", 90000, 0, "INDIVIDUAL", "RENT",
            15000, " 60 months", 0.125, "C", "Mar-16", "debt_consolidation", 400, 8000, 20000, "Entry Level"
        ]
        with open(NEW_APPLICATIONS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(dummy_data_row)
            writer.writerow(["C3", "Manager", "TX", 120000, 50000, "JOINT", "OWN", 25000, " 36 months", 0.082, "A", "Apr-16", "home_improvement", 780, 10000, 30000, "Entry Level"])


    print(f"Connecting to RabbitMQ on host '{RABBITMQ_HOST}'...")
    try:
        connection_params = pika.ConnectionParameters(host=RABBITMQ_HOST)
        connection = pika.BlockingConnection(connection_params)
        channel = connection.channel()

        # Declare a durable queue
        channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
        print(f"Declared queue '{RABBITMQ_QUEUE}'.")

    except pika.exceptions.AMQPConnectionError as e:
        print(f"Failed to connect to RabbitMQ: {e}")
        print("Please ensure RabbitMQ server is running and accessible.")
        return

    print(f"Starting to read from '{NEW_APPLICATIONS_CSV}' and send to queue '{RABBITMQ_QUEUE}'.")
    try:
        with open(NEW_APPLICATIONS_CSV, mode='r', newline='') as file:
            csv_reader = csv.DictReader(file)
            for row_index, row_data in enumerate(csv_reader):
                # Basic type conversion for known numeric fields
                for key in ['annual_inc', 'annual_inc_joint', 'loan_amount', 'int_rate', 'installment', 'avg_cur_bal', 'tot_cur_bal']:
                    if row_data.get(key) is not None and row_data.get(key) != '':
                        try:
                            row_data[key] = float(row_data[key])
                        except ValueError:
                            print(f"Warning: Could not convert '{row_data[key]}' to float for column '{key}' in row {row_index+1}. Keeping as string or None.")
                            if row_data[key] == '':
                                row_data[key] = None 
                    elif row_data.get(key) == '':
                         row_data[key] = None


                send_message_to_queue(channel, row_data)
                time.sleep(1)  # Simulate applications arriving over time

    except FileNotFoundError:
        print(f"Error: The file {NEW_APPLICATIONS_CSV} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'connection' in locals() and connection.is_open:
            connection.close()
            print("RabbitMQ connection closed.")

if __name__ == '__main__':
    main()
