import urllib.request,json,pathlib
u='https://robotwin-platform.github.io/data/robotwin_leaderboard.json'
d=json.load(urllib.request.urlopen(u,timeout=30));pathlib.Path('.local/flux-piperx-20261008/public-robotwin-leaderboard.json').write_text(json.dumps(d,ensure_ascii=False),encoding='utf-8');print(type(d).__name__);print(list(d) if isinstance(d,dict) else len(d))
