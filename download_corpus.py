import os
import requests

os.makedirs('data', exist_ok=True)

TARGET_MB = 6 
target_bytes = TARGET_MB * 1024 * 1024
total_bytes = 0
saved_poems = 0

def is_stanza_poem(text): #checks if the text is written in verse 
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    
    # if it's empty or too short to be useful, skip it
    if len(lines) < 10:
        return False
        
    # checks if it's a verse? Average line length should be short (under 80 chars)
    avg_line_length = sum(len(line) for line in lines) / len(lines)
    if avg_line_length > 80: 
        return False
        
    # is it a drama? (Filter out texts with lots of ALL CAPS character names like "GUSTAW:" becuase there was a lot of those at first try)
    if sum(1 for line in lines if line.isupper()) > len(lines) * 0.05:
        return False

    return True
    
response = requests.get('https://wolnelektury.pl/api/kinds/liryka/books/')
books = response.json()

print(f"Found {len(books)} potential poems. Hunting for {TARGET_MB} MB of stanzas...\n")

for book in books:
    if total_bytes >= target_bytes:
        break

    slug = book.get('slug')
    if not slug:
        continue
        
    print(f"Checking: {slug[:30]:<30} ... ", end="", flush=True)
    
    txt_url = f"https://wolnelektury.pl/media/book/txt/{slug}.txt"

    try:
        text_response = requests.get(txt_url, timeout=5)  # added a 5-second timeout so it doesn't hang on a bad connection
        
        if text_response.status_code != 200:
            print("Skipped (No text file found)")
            continue
            
        raw_text = text_response.text
        clean_text = raw_text.split("Ta lektura, podobnie jak tysiące innych")[0].strip()
        
        if is_stanza_poem(clean_text):
            filename = f"data/{slug}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(clean_text)
            
            size = os.path.getsize(filename)
            total_bytes += size
            saved_poems += 1
            print(f"SAVED! (Total: {total_bytes / 1024 / 1024:.2f} MB)")
        else:
            print("Skipped (Did not pass stanza filter)")
            
    except Exception as e:
        print(f"Error checking file.")
        continue

print(f"\n=============================================")
print(f"Success! Downloaded {saved_poems} perfectly structured poems.")
print(f"Final Size: {total_bytes / 1024 / 1024:.2f} MB")
print(f"=============================================")
