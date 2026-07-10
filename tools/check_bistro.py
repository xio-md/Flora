import os, sys, json
g = json.load(open(r'D:\niagara_bistro\bistro.gltf', encoding='utf-8'))
count = 0
for img in g.get('images', []):
    uri = img.get('uri', '')
    if uri and count < 5:
        full = os.path.join(r'D:\niagara_bistro', uri.replace('/', os.sep))
        print(f'  {uri} -> exists={os.path.exists(full)}')
        count += 1
print(f'Total images: {len(g.get("images", []))}')
# Check extensions info
ext = g.get('extensionsUsed', [])
print(f'Extensions used: {ext}')
