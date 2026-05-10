"""
Verify all services are running correctly before starting the app.
Usage: python check_services.py
"""
import sys

def check(name, fn):
    try:
        fn()
        print(f"  ✅ {name}")
        return True
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return False

print("\n🔍 Checking services...\n")

results = []

# PostgreSQL
def check_postgres():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from sqlalchemy import create_engine, text
    engine = create_engine(os.getenv("DATABASE_URL"))
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

results.append(check("PostgreSQL (port 5432)", check_postgres))

# Qdrant
def check_qdrant():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from qdrant_client import QdrantClient
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333"))
    )
    client.get_collections()

results.append(check("Qdrant (port 6333)", check_qdrant))

# Neo4j
def check_neo4j():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password123"))
    )
    with driver.session() as s:
        s.run("RETURN 1")
    driver.close()

results.append(check("Neo4j (port 7687)", check_neo4j))

# Gemini API Key
def check_gemini():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY", "")
    if not key or key == "your-gemini-api-key-here":
        raise ValueError("GEMINI_API_KEY not set in .env")

results.append(check("Gemini API key", check_gemini))

print()
passed = sum(results)
total = len(results)

if passed == total:
    print(f"✅ All {total} services ready! You can start the app.\n")
else:
    print(f"⚠️  {passed}/{total} services ready.")
    print("   Start missing services with: docker-compose up -d\n")
    sys.exit(1)
