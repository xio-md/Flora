import json
g = json.load(open(r'D:\niagara_bistro\bistro.gltf', encoding='utf-8'))
# First texture
t = g['textures'][0]
print('Texture 0 extensions:', json.dumps(t.get('extensions', {}), indent=2))
i = t['source']
print(f'Source image: index={i}, uri={g["images"][i]["uri"]}')

# Check if DDS extension has a source pointing to existing image
ext = t.get('extensions', {}).get('MSFT_texture_dds', {})
if 'source' in ext:
    dds_idx = ext['source']
    print(f'MSFT_texture_dds source: index={dds_idx}, uri={g["images"][dds_idx]["uri"]}')
