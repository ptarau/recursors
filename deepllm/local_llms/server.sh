python3 -m fastchat.serve.controller &
sleep 10
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.3 &
sleep 10
python3 -m fastchat.serve.openai_api_server --host u.local --port 8000


