import os

root_dir = "src/trust_agents"
old_import = "fake_news_detector"
new_import = "trust_agents"

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if old_import in content:
                new_content = content.replace(old_import, new_import)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"Updated: {path}")
