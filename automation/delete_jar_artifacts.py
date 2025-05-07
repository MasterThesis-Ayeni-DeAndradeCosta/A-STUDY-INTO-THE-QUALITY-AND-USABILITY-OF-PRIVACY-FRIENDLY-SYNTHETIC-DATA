import os

def delete_jar_files():
    # Target the correct jar location used in your runner code
    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src", "anonymization", "anonymARXij", "bin")
    )
    print(f"📂 Scanning for .jar files in: {base_path}")

    if not os.path.exists(base_path):
        print("❌ Path does not exist!")
        return

    found = False
    for root, dirs, files in os.walk(base_path):
        for file in files:
            print(f"🔍 Found file: {file}")
            if file.lower().endswith(".jar"):
                jar_path = os.path.join(root, file)
                try:
                    os.remove(jar_path)
                    found = True
                    print(f"🗑️ Deleted JAR file: {jar_path}")
                except Exception as e:
                    print(f"❌ Failed to delete {jar_path}. Reason: {e}")

    if not found:
        print("⚠️ No JAR files found to delete.")

if __name__ == "__main__":
    delete_jar_files()
