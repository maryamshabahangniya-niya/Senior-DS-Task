
#Libraries
import os
import psycopg2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt





#make directory
os.makedirs("charts/eda", exist_ok=True)





#DB config
DB_HOST = "db"
DB_PORT = 5432
DB_NAME = "leads_db"
DB_USER = "admin"
DB_PASSWORD = "admin"







#connet DB and Load Data
conn = psycopg2.connect(
    host=DB_HOST, 
    port=DB_PORT, 
    dbname=DB_NAME, 
    user=DB_USER, 
    password=DB_PASSWORD)


df = pd.read_sql_query("SELECT * FROM leads", conn)
conn.close()










#base information
print("\nShape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum().sort_values(ascending=False))
print("\nDuplicate rows:", df.duplicated().sum())
print("\nDuplicate Lead IDs:", df["Lead ID"].duplicated().sum())
print("\nTarget distribution (%):\n", df["Completed Purchase"].value_counts(normalize=True) * 100)
print("\nNumeric summary:\n", df.describe(include="number").T)







column_summary = pd.DataFrame({
    "Data Type": df.dtypes,
    "Unique Values": df.nunique(),
    "Missing Count": df.isnull().sum(),
    "Missing Percent": df.isnull().mean() * 100
})





print("\nColumn Summary:")
print(column_summary.to_string())


categorical_columns = [
    "Product Type", 
    "Channel", "Device", 
    "Partner","City", 
    "Insurance Company", "Payment Type"
]





for col in categorical_columns:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).to_string())






# Check duplicated lead IDs
duplicate_leads = df[df.duplicated(subset="Lead ID", keep=False)].sort_values("Lead ID")

print("\nDuplicate Lead ID rows:", duplicate_leads.shape)
print("\nRecords per duplicated Lead ID:")
print(duplicate_leads["Lead ID"].value_counts().value_counts().sort_index())
print("\nDuplicate rows summary:\n", duplicate_leads.describe(include="all").T)



different_columns = {}

for col in df.columns:
    if col == "Lead ID":
        continue

    count = duplicate_leads.groupby("Lead ID")[col].nunique(dropna=False).gt(1).sum()
    different_columns[col] = count

print("\nDifferent columns among duplicate Lead IDs:")
print(pd.Series(different_columns).sort_values(ascending=False).to_string())







# Keep the earliest record for each lead
df = df.sort_values("Created At").drop_duplicates(subset="Lead ID", keep="first").copy()
print("\nShape after removing duplicate Lead IDs:", df.shape)








# Time features
df["Created Hour"] = df["Created At"].dt.hour
df["Created DayOfWeek"] = df["Created At"].dt.day_name()
df["Created Period"] = pd.cut(df["Created Hour"], bins=[-1, 5, 11, 17, 23], labels=["Night", "Morning", "Afternoon", "Evening"])







expiry_corr = df["Days To Policy Expiry"].corr(df["Completed Purchase"])
print("\nCorrelation between Policy Expiry and Purchase:", expiry_corr)





# Price outlier check using IQR
price_summary = df.groupby("Product Type")["Price"].agg(["count", "min", "median", "mean", "max"])

print("\nPrice Summary by Product Type:")
print(price_summary)

price_outlier_results = []

for product_type, group in df.groupby("Product Type"):

    q1 = group["Price"].quantile(0.25)
    q3 = group["Price"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = group["Price"].notna() & ((group["Price"] < lower_bound) | (group["Price"] > upper_bound))

    price_outlier_results.append({
        "Product Type": product_type,
        "Total Leads": len(group),
        "Valid Prices": group["Price"].notna().sum(),
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": outliers.sum(),
        "Outlier Percent": outliers.mean() * 100
    })

price_outlier_summary = pd.DataFrame(price_outlier_results)

print("\nPrice Outlier Analysis by Product Type:")
print(price_outlier_summary.to_string(index=False))







print("\nExpected Margin by Product Type:")
print(df.groupby("Product Type")["Expected Margin"].describe())




# --------------------------------
# Plots
# --------------------------------

# 1.Target distribution
df["Completed Purchase"].value_counts().sort_index().plot(kind="bar")
plt.title("Target Distribution")
plt.xlabel("Completed Purchase")
plt.ylabel("Number of Leads")
plt.tight_layout()
plt.savefig("charts/eda/target_distribution.png")
plt.close()






# 2.Conversion by channel
conversion_channel = df.groupby("Channel")["Completed Purchase"].mean().sort_values(ascending=False) * 100

conversion_channel.plot(kind="bar")
plt.title("Conversion Rate by Channel")
plt.xlabel("Channel")
plt.ylabel("Conversion Rate (%)")
plt.tight_layout()
plt.savefig("charts/eda/conversion_by_channel.png")
plt.close()







# 3.Conversion by product type
conversion_product = df.groupby("Product Type")["Completed Purchase"].mean().sort_values(ascending=False) * 100

conversion_product.plot(kind="bar")
plt.title("Conversion Rate by Product Type")
plt.xlabel("Product Type")
plt.ylabel("Conversion Rate (%)")
plt.tight_layout()
plt.savefig("charts/eda/conversion_by_product.png")
plt.close()







# 4.Price distribution
df.boxplot(column="Price", by="Completed Purchase")
plt.title("Price Distribution by Purchase")
plt.suptitle("")
plt.xlabel("Completed Purchase")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("charts/eda/price_by_purchase.png")
plt.close()









# 5.Conversion by city
city_analysis = df.groupby("City").agg(
    Total_Leads=("Lead ID", "count"),
    Total_Purchases=("Completed Purchase", "sum"),
    Conversion_Rate=("Completed Purchase", "mean")
)

city_analysis["Conversion_Rate"] *= 100
city_analysis = city_analysis.sort_values("Conversion_Rate", ascending=False)

print("\nConversion Rate by City:\n", city_analysis)

city_analysis["Conversion_Rate"].plot(kind="bar")
plt.title("Conversion Rate by City")
plt.xlabel("City")
plt.ylabel("Conversion Rate (%)")
plt.tight_layout()
plt.savefig("charts/eda/conversion_by_city.png")
plt.close()


# 6.Conversion by policy expiry
df["Expiry_Group"] = pd.cut(
    df["Days To Policy Expiry"],
    bins=[-float("inf"), -1, 7, 15, 30, float("inf")],
    labels=["Expired", "0-7 Days", "8-15 Days", "16-30 Days", "30+ Days"]
)

expiry_analysis = df.groupby("Expiry_Group", observed=True).agg(
    Total_Leads=("Lead ID", "count"),
    Total_Purchases=("Completed Purchase", "sum"),
    Conversion_Rate=("Completed Purchase", "mean")
)

expiry_analysis["Conversion_Rate"] *= 100

print("\nConversion Rate by Policy Expiry:\n", expiry_analysis)

expiry_analysis["Conversion_Rate"].plot(kind="bar")
plt.title("Conversion Rate by Days to Policy Expiry")
plt.xlabel("Days to Policy Expiry")
plt.ylabel("Conversion Rate (%)")
plt.tight_layout()
plt.savefig("charts/eda/conversion_by_expiry.png")
plt.close()






# 7.Conversion by insurance company
insurance_analysis = df.groupby("Insurance Company").agg(
    Total_Leads=("Lead ID", "count"),
    Total_Purchases=("Completed Purchase", "sum"),
    Conversion_Rate=("Completed Purchase", "mean")
)

insurance_analysis["Conversion_Rate"] *= 100
insurance_analysis = insurance_analysis.sort_values("Conversion_Rate", ascending=False)

print("\nConversion Rate by Insurance Company:\n", insurance_analysis)

insurance_analysis["Conversion_Rate"].plot(kind="bar")
plt.title("Conversion Rate by Insurance Company")
plt.xlabel("Insurance Company")
plt.ylabel("Conversion Rate (%)")
plt.tight_layout()
plt.savefig("charts/eda/conversion_by_insurance.png")
plt.close()









# 8.Conversion by partner
partner_analysis = df.groupby("Partner").agg(
    Total_Leads=("Lead ID", "count"),
    Total_Purchases=("Completed Purchase", "sum"),
    Conversion_Rate=("Completed Purchase", "mean")
)

partner_analysis["Conversion_Rate"] *= 100
partner_analysis = partner_analysis.sort_values("Conversion_Rate", ascending=False)

print("\nConversion Rate by Partner:\n", partner_analysis)

partner_analysis["Conversion_Rate"].plot(kind="bar")
plt.title("Conversion Rate by Partner")
plt.xlabel("Partner")
plt.ylabel("Conversion Rate (%)")
plt.tight_layout()
plt.savefig("charts/eda/conversion_by_partner.png")
plt.close()









# 9.Previous purchase
previous_purchase_analysis = df.groupby("Has Previous Purchase").agg(
    Total_Leads=("Lead ID", "count"),
    Total_Purchases=("Completed Purchase", "sum"),
    Conversion_Rate=("Completed Purchase", "mean")
)

previous_purchase_analysis["Conversion_Rate"] *= 100

print("\nPrevious Purchase Analysis:\n", previous_purchase_analysis)

previous_purchase_analysis["Conversion_Rate"].plot(kind="bar")
plt.title("Conversion Rate by Previous Purchase")
plt.xlabel("Has Previous Purchase")
plt.ylabel("Conversion Rate (%)")
plt.tight_layout()
plt.savefig("charts/eda/conversion_by_previous_purchase.png")
plt.close()








# 10.Correlation with target
numeric_df = df.select_dtypes(include="number")
target_corr = numeric_df.corr()["Completed Purchase"].drop("Completed Purchase").sort_values(ascending=False)

print("\nCorrelation with Completed Purchase:\n", target_corr)

target_corr.sort_values().plot(kind="barh", figsize=(10, 7))
plt.title("Correlation with Completed Purchase")
plt.xlabel("Pearson Correlation")
plt.tight_layout()
plt.savefig("charts/eda/target_correlation.png")
plt.close()









# 11.Conversion by time period
time_analysis = df.groupby("Created Period", observed=True).agg(
    Total_Leads=("Lead ID", "count"),
    Total_Purchases=("Completed Purchase", "sum"),
    Conversion_Rate=("Completed Purchase", "mean")
)

time_analysis["Conversion_Rate"] *= 100

print("\nConversion by Time Period:\n", time_analysis)

time_analysis["Conversion_Rate"].plot(kind="bar")
plt.title("Conversion Rate by Time Period")
plt.ylabel("Conversion Rate (%)")
plt.tight_layout()
plt.savefig("charts/eda/conversion_by_time_period.png")
plt.close()








# 12.Conversion by day of week
day_analysis = df.groupby("Created DayOfWeek").agg(
    Total_Leads=("Lead ID", "count"),
    Total_Purchases=("Completed Purchase", "sum"),
    Conversion_Rate=("Completed Purchase", "mean")
)

day_analysis["Conversion_Rate"] *= 100
day_order = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
day_analysis = day_analysis.reindex(day_order)

print("\nConversion by Day of Week:\n", day_analysis)

day_analysis["Conversion_Rate"].plot(kind="bar")
plt.title("Conversion Rate by Day of Week")
plt.ylabel("Conversion Rate (%)")
plt.tight_layout()
plt.savefig("charts/eda/conversion_by_day_of_week.png")
plt.close()








# 13.Conversion by hour
hour_analysis = df.groupby("Created Hour").agg(
    Total_Leads=("Lead ID", "count"),
    Total_Purchases=("Completed Purchase", "sum"),
    Conversion_Rate=("Completed Purchase", "mean")
)

hour_analysis["Conversion_Rate"] *= 100

print("\nConversion by Hour:\n", hour_analysis)

hour_analysis["Conversion_Rate"].plot(kind="line", marker="o", figsize=(10, 5))
plt.title("Conversion Rate by Created Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Conversion Rate (%)")
plt.xticks(range(24))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("charts/eda/conversion_by_hour.png")
plt.close()








# 14. BoxPlot for price column in different product Type
df.boxplot(column="Price", by="Product Type")
plt.title("Price Distribution by Product Type")
plt.suptitle("")
plt.xlabel("Product Type")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("charts/eda/price_by_product_type.png")
plt.close()





print("\nEDA charts saved in charts/eda/")
