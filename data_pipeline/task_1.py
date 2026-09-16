import sqlite3
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

base_url = "http://books.toscrape.com/catalogue/"
GBP_to_INR_Rate = 105.50
db_file = "books_catalog.db"
query_output_file = "query_results.txt"

rating_map = {"One": 1,"Two": 2,"Three": 3,"Four": 4,"Five": 5}

def scrap_books(target_count=69):
    scraped_data = []
    page = 1
    
    while len(scraped_data) < target_count:
        url = f"http://books.toscrape.com/catalogue/page-{page}.html"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            break
        
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("article", class_="product_pod")
        
        if not articles:
            break
            
        for article in articles:
            title = article.h3.a["title"].strip()
            price_text = article.find("p", class_="price_color").text.strip()

            rating_p = article.find("p", class_="star-rating")
            rating_classes = rating_p.get("class", [])
            rating_text = [c for c in rating_classes if c != "star-rating"]
            rating_str = rating_text[0] if rating_text else None

            availability_text = article.find("p", class_="instock availability").text.strip()
            
            detail_url = base_url + article.h3.a["href"]
            detail_response = requests.get(detail_url, timeout=10)
            
            if detail_response.status_code == 200:
                detail_soup = BeautifulSoup(detail_response.text, "html.parser")
                breadcrumb = detail_soup.find("ul", class_="breadcrumb")
                crumbs = breadcrumb.find_all("li")
                category_name = crumbs[2].text.strip() if len(crumbs) >= 3 else "General"
            else:
                category_name = "General"
            
            scraped_data.append({
                "title": title,
                "price_raw": price_text,
                "star_rating_raw": rating_str,
                "availability_raw": availability_text,
                "category": category_name
            })
            
            if len(scraped_data) >= target_count:
                break
                
        page += 1

    return pd.DataFrame(scraped_data)

def clean_and_transform(df):
    def parse_price(val):
        match = re.search(r"(\d+\.\d+)", str(val))
        return float(match.group(1)) if match else None

    df["price_gbp"] = df["price_raw"].apply(parse_price)
    if df["price_gbp"].isnull().any():
        median_price = df["price_gbp"].median()
        df["price_gbp"].fillna(median_price, inplace=True)

    df["rating"] = df["star_rating_raw"].map(rating_map).fillna(0).astype(int)

    df["in_stock"] = df["availability_raw"].str.contains("In stock", case=False, na=False)

    df["price_inr"] = (df["price_gbp"] * GBP_to_INR_Rate).round(2)

    return df

def load_to_db(df):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS books;")
    cursor.execute("DROP TABLE IF EXISTS categories;")

    cursor.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    );
    """)

    unique_categories = df["category"].dropna().unique()
    for cat in unique_categories:
        cursor.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?);", (str(cat),))
    conn.commit()

    category_df = pd.read_sql_query("SELECT category_id, category_name FROM categories;", conn)
    cat_map = dict(zip(category_df["category_name"], category_df["category_id"]))
    df["category_id"] = df["category"].map(cat_map).astype(int)
    df["in_stock"] = df["in_stock"].astype(int)

    cols = ["title", "price_gbp", "price_inr", "rating", "in_stock", "category_id"]
    records = list(df[cols].itertuples(index=False, name=None))

    cursor.executemany("""
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?, ?);
    """, records)
    conn.commit()

    return conn

def execute_queries(conn):
    queries = {
        "Query 1 (SELECT, WHERE, ORDER BY, LIMIT)": """
            SELECT title, price_inr, rating 
            FROM books 
            WHERE in_stock = 1 
            ORDER BY price_inr DESC 
            LIMIT 5;
        """,
        "Query 2 (DISTINCT)": """
            SELECT DISTINCT rating 
            FROM books 
            ORDER BY rating ASC;
        """,
        "Query 3 (BETWEEN)": """
            SELECT title, price_gbp 
            FROM books 
            WHERE price_gbp BETWEEN 20.00 AND 30.00 
            ORDER BY price_gbp ASC 
            LIMIT 5;
        """,
        "Query 4 (IN)": """
            SELECT title, rating 
            FROM books 
            WHERE rating IN (1, 5) 
            LIMIT 5;
        """,
        "Query 5 (JOIN - Top rated books per category)": """
            SELECT b.title, c.category_name, b.rating, b.price_inr
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
            WHERE b.rating = 5
            ORDER BY c.category_name ASC, b.price_inr DESC
            LIMIT 5;
        """
    }

    with open(query_output_file, "w") as f:
           for name, q in queries.items():
               result = pd.read_sql_query(q, conn)
   
               block = f"\n{name}\nSQL:\n{q.strip()}\n\nResult:\n{result.to_string(index=False)}\n"
               print(block)
               f.write(block)
   
    return queries["Query 5 (JOIN - Top rated books per category)"]

def verify_sql_vs_pandas(conn, join_query):
    print("\nVERIFYING pd.read_sql vs pd.merge")
   
    df_sql_result = pd.read_sql(join_query, conn)
    
    df_books = pd.read_sql("SELECT title, price_inr, rating, category_id FROM books;", conn)
    df_categories = pd.read_sql("SELECT category_id, category_name FROM categories;", conn)

    df_books["category_id"] = pd.to_numeric(df_books["category_id"]).astype(int)
    df_categories["category_id"] = pd.to_numeric(df_categories["category_id"]).astype(int)

    df_merged = pd.merge(df_books, df_categories, on="category_id")
    df_merged = df_merged[df_merged["rating"] == 5]
    df_merged = df_merged[["title", "category_name", "rating", "price_inr"]]
    df_merged = df_merged.sort_values(by=["category_name", "price_inr"], ascending=[True, False]).head(5)
    df_merged = df_merged.reset_index(drop=True)

    print("\n[Output via pd.read_sql]")
    print(df_sql_result)

    print("\n[Output via in-memory pd.merge]")
    print(df_merged)

    matches = df_sql_result.equals(df_merged)
    print(f"\nDataFrames match exactly: {matches}")

if __name__ == "__main__":
    print("Scraping books from books.toscrape.com...")
    raw_df = scrap_books(target_count=69)
    print(f"Scraped {len(raw_df)} books across {raw_df['category'].nunique()} categories.")

    cleaned_df = clean_and_transform(raw_df)
    
    conn = load_to_db(cleaned_df)
    
    join_q = execute_queries(conn)
    verify_sql_vs_pandas(conn, join_q)
    
    conn.close()
    print(f"\nPipeline execution complete. Database saved as books_catalog.db.  , query output saved to {query_output_file}")
    
    
    
    