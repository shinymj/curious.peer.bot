# curious.peer.bot
 supporting critical thinking

### Heavily assisted by Claude-3.5-sonnet

# Structure

curious.peer.bot/   
├── .env   
├── requirements.txt   
├── main.py   
├── bot_src/   
│   ├── __init__.py   
│   ├── sys_prompt.txt   
│   ├── bot.py   
│   └── utils.py   
└── _output/   

# Notes
Updating library `pip install -r requirements.txt --upgrade`

# Version log
`main.py`
- 'type' parameter 관련된 deprecation warning 해결
- 업로드 파일명 저장 안되는 문제 해결 
- submit method (ctrl+enter 로 submit) 작동 안함

`main_00.py`
- 돌아가긴 함. 
- 'type' parameter 관련된 deprecation warning 있음. 
- 업로드한 파일명이 저장되지 않는 문제 있음. submit method (ctrl+enter 로 submit) 작동 안함