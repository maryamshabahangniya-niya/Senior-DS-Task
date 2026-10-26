#Libraries
import psycopg2


# Database settings
DB_HOST = "db"
DB_PORT = 5432
DB_NAME = "leads_db"
DB_USER = "admin"
DB_PASSWORD = "admin"




# File settings
CSV_PATH = "data/leads.csv"
TABLE_NAME = "leads"

# Connect to PostgreSQL
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

cursor = conn.cursor()

# Create table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS leads (
        "Lead ID" TEXT,
        "Created At" TIMESTAMP,
        "Product Type" TEXT,
        "Channel" TEXT,
        "Device" TEXT,
        "Partner" TEXT,
        "City" TEXT,
        "Insurance Company" TEXT,
        "Payment Type" TEXT,
        "Minutes Since Abandonment" INTEGER,
        "Days To Policy Expiry" INTEGER,
        "Price" NUMERIC,
        "Discount Percent" NUMERIC,
        "Has Previous Purchase" INTEGER,
        "Visited Offer Page" INTEGER,
        "Incoming Call Last 24h" INTEGER,
        "Sessions Last 7d" INTEGER,
        "Offer Views Last 7d" INTEGER,
        "Price Comparisons Last 7d" INTEGER,
        "Days Since Last Visit" NUMERIC,
        "Expected Margin" NUMERIC,
        "Completed Purchase" INTEGER
    )
""")

# Clear previous data
cursor.execute("TRUNCATE TABLE leads")

# Import CSV
with open(CSV_PATH, "r", encoding="utf-8-sig") as csv_file:
    cursor.copy_expert("COPY leads FROM STDIN WITH CSV HEADER", csv_file)





# Check number of imported rows
cursor.execute("SELECT COUNT(*) FROM leads")
row_count = cursor.fetchone()[0]

conn.commit()

cursor.close()
conn.close()

print("Imported rows:", row_count)
