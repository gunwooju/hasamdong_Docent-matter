import gradio as gr
import pandas as pd
import embedding
import json
import os
import random
import base64
import re
import speech_recognition as sr
from pydub import AudioSegment
from pydub.effects import normalize
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage
from langchain_upstage import ChatUpstage
from tools import similar_art_search, qa_with_explain, empathize_with_user, normal_chat, wiki_search, archiving
from langchain_core.output_parsers import StrOutputParser
from gtts import gTTS
import tempfile


# 환경 변수 로드
load_dotenv()

# ChatUpstage 모델 초기화
llm = ChatUpstage(streaming=True)

# 사용할 도구 목록 정의
tools = [similar_art_search, qa_with_explain, empathize_with_user, normal_chat, wiki_search, archiving]
llm_with_tools = llm.bind_tools(tools)

# 데이터 불러오기
bokcheon_df = pd.read_csv("data/bokcheon_museum.csv")
tal_df = pd.read_csv("data/tal_museum.csv")

# 출처 컬럼 추가
bokcheon_df['source'] = 'bokcheon'
tal_df['source'] = 'tal'

# DB 및 Retrieval 불러오기
searching_bokcheon = embedding.Search(bokcheon_df)
searching_tal = embedding.Search(tal_df)

def convert_to_wav(m4a_file_path):
    """M4A 파일을 WAV로 변환합니다."""
    audio = AudioSegment.from_file(m4a_file_path, format="m4a")
    wav_file_path = m4a_file_path.replace(".m4a", ".wav")
    audio.export(wav_file_path, format="wav")
    return wav_file_path

def reduce_noise(audio_file_path):
    """소음을 줄이고 오디오를 정규화합니다."""
    audio = AudioSegment.from_file(audio_file_path, format="wav")
    normalized_audio = normalize(audio)
    normalized_audio.export(audio_file_path, format="wav")

def transcribe_audio(audio_file_path, chunk_length=60000):
    """오디오 파일을 텍스트로 변환합니다."""
    if not audio_file_path:
        return "[음성 파일이 지정되지 않았습니다.]"
    
    recognizer = sr.Recognizer()
    
    if audio_file_path.endswith('.m4a'):
        audio_file_path = convert_to_wav(audio_file_path)
    
    reduce_noise(audio_file_path)
    
    audio = AudioSegment.from_file(audio_file_path, format="wav")
    
    if len(audio) < 1000:
        return "[음성 파일이 너무 짧습니다. 최소 1초 이상의 파일을 업로드 해주세요.]"
    elif len(audio) > 600000:
        return "[음성 파일이 너무 깁니다. 최대 10분 이하의 파일을 업로드 해주세요.]"
    
    chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
    full_text = ""
    
    for i, chunk in enumerate(chunks):
        if len(chunk) < 1000:
            continue
        
        chunk_file_path = f"chunk_{i}.wav"
        chunk.export(chunk_file_path, format="wav")
        
        with sr.AudioFile(chunk_file_path) as source:
            audio_data = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_data, language='ko-KR')
            full_text += text + " "
        except sr.UnknownValueError:
            full_text += "[음성을 인식할 수 없습니다.] "
        except sr.RequestError as e:
            full_text += f"[음성 인식 서비스에 접근할 수 없습니다. 에러: {e}] "
        finally:
            os.remove(chunk_file_path)
    
    if audio_file_path.endswith('.wav'):
        os.remove(audio_file_path)
    
    return full_text

def handle_audio(file, history, cur_art):
    if not file:
        return "", history, cur_art
     
    text = transcribe_audio(file)
    if text is None:
        text = "[음성 인식에 실패했습니다.]"
    
    updated_history = chat(history + [[text, None]], cur_art)
    return "", updated_history, cur_art

def get_image_path(row):
    """각 작품의 이미지 경로를 반환하는 함수"""
    if row['source'] == 'bokcheon':
        base_path = "data/bok_images"
    else:
        base_path = "data/tal_images"
    
    image_number = str(row['이미지']).split('.')[0]
    
    extensions = ['.jpg', '.jpeg', '.png', '.gif']
    
    for ext in extensions:
        full_path = os.path.join(base_path, f"{image_number}{ext}")
        if os.path.exists(full_path):
            print(f"이미지를 찾았습니다: {full_path}")
            return full_path
    
    print(f"이미지를 찾을 수 없습니다: {row['유물명'] if '유물명' in row else row['작품명']} (번호: {image_number})")
    print(f"확인한 경로: {[os.path.join(base_path, f'{image_number}{ext}') for ext in extensions]}")
    return None

def convert_base64(image_path):
    """이미지를 base64로 인코딩하는 함수"""
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    return None

def get_all_artworks():
    """모든 작품 목록을 반환하는 함수"""
    bokcheon_list = [f"{row['유물명']} ({row['시대']})" for _, row in bokcheon_df.iterrows()]
    tal_list = [f"{row['작품명']} ({row['제작년도']})" for _, row in tal_df.iterrows()]
    all_list = bokcheon_list + tal_list
    
    search_dict = {}
    for idx, item in enumerate(all_list):
        if idx < len(bokcheon_list):
            search_dict[item] = {'index': bokcheon_df.iloc[idx]['번호'], 'source': 'bokcheon'}
        else:
            search_dict[item] = {'index': tal_df.iloc[idx - len(bokcheon_list)]['번호'], 'source': 'tal'}
    
    return all_list, json.dumps(search_dict, ensure_ascii=False)

def search_art(query, selected_museum):
    """선택된 박물관에서 작품을 검색하는 함수"""
    if selected_museum == "복천 박물관":
        results = searching_bokcheon.search(query, searching_bokcheon.retriever_full)
    elif selected_museum == "탈 박물관":
        results = searching_tal.search(query, searching_tal.retriever_full)
    else:
        results_bokcheon = searching_bokcheon.search(query, searching_bokcheon.retriever_full)
        results_tal = searching_tal.search(query, searching_tal.retriever_full)
        results = results_bokcheon + results_tal
    
    search_list = []
    search_dict = {}
    
    for result in results:
        element = f"{result.metadata['title']} ({result.metadata['year']})"
        search_list.append(element)
        search_dict[element] = result.metadata
        search_dict[element]['source'] = 'bokcheon' if selected_museum == "복천 박물관" or (selected_museum == "모두" and result in results_bokcheon) else 'tal'
    
    return gr.update(choices=search_list, value=None), json.dumps(search_dict, ensure_ascii=False)

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='ko')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        print(f"TTS 에러: {str(e)}")
        return None

def dropdown_change(item, search_dict, is_child, generate_audio):
    if item:
        search_dict = json.loads(search_dict)
        if search_dict[item].get('source') == 'bokcheon':
            search_art = bokcheon_df[bokcheon_df['번호'].astype(str) == str(search_dict[item]['index'])]
        else:
            search_art = tal_df[tal_df['번호'].astype(str) == str(search_dict[item]['index'])]
        
        if search_art.empty:
            print(f"작품을 찾을 수 없습니다: {item}")
            return None, None, "작품을 찾을 수 없습니다.", None, gr.update(visible=False, value=None)

        search_art = search_art.iloc[0]
        
        image_path = get_image_path(search_art)
        if image_path:
            if search_dict[item].get('source') == 'bokcheon':
                description = search_art['어린이 전용'] if is_child else search_art['작품 설명']
            else:
                description = search_art['어린아이'] if is_child else search_art['작품설명']
            
            audio_path = None
            if generate_audio:
                audio_path = text_to_speech(description)
                if audio_path is None:
                    print("음성 생성에 실패했습니다.")
            
            return image_path, search_art.to_json(force_ascii=False), description, audio_path, gr.update(visible=generate_audio, value=audio_path)
        else:
            return None, search_art.to_json(force_ascii=False), "이미지를 찾을 수 없습니다.", None, gr.update(visible=False, value=None)
    else:
        return None, None, None, None, gr.update(visible=False, value=None)
def user(user_message, history, cur_art):
    """사용자 메시지를 처리하는 함수"""
    return "", history + [[user_message, None]], cur_art

def chat(history, cur_art):
    """AI와의 대화를 처리하는 함수"""
    if not history:
        history = []

    # Extract the last message from history
    message = history[-1][0] if history else ""

    # Check if the message is empty
    if not message.strip():
        print("Error: Message is empty.")
        return history

    # Load current artwork information
    cur_art = cur_art if isinstance(cur_art, str) else cur_art.value if hasattr(cur_art, 'value') else "{}"
    cur_art = json.loads(cur_art) if cur_art else {}

    basic_prompt = f"""
    ## Role: 박물관 도슨트

    ## Instruction
    - 주어진 작품 정보를 기반으로 사용자와 대화합니다.
    - 작품에 대한 정보를 정확하고 이해하기 쉽게 설명합니다.
    - 사용자의 질문에 대해 친절하고 상세하게 답변합니다.
    - 탈 박물관의 작품일 경우, 탈의 특징과 의미에 대해 설명합니다.

    ## 현재 작품 정보
    유물명/작품명: {cur_art.get('유물명') or cur_art.get('작품명')}
    시대/제작년도: {cur_art.get('시대') or cur_art.get('제작년도')}
    발굴 장소/재료: {cur_art.get('발굴 장소') or cur_art.get('재료', '정보 없음')}
    규격: {cur_art.get('규격', '정보 없음')}
    작품 설명: {cur_art.get('작품 설명') or cur_art.get('작품설명')}
    간단 요약: {cur_art.get('간단 요약') or cur_art.get('간단요약', '정보 없음')}
    어린이 전용: {cur_art.get('어린이 전용') or cur_art.get('어린아이', '정보 없음')}
    박물관 출처: {'탈 박물관' if cur_art.get('source') == 'tal' else '복천 박물관'}
    """

    chat_with_history_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", basic_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{message}"),
        ]
    )

    chain = chat_with_history_prompt | llm | StrOutputParser()

    history_langchain_format = [AIMessage(content=basic_prompt)]
    for human, ai in history[:-1]:
        history_langchain_format.append(HumanMessage(content=human))
        if ai:
            history_langchain_format.append(AIMessage(content=ai))

    # Use RAG if applicable, otherwise continue with the normal process
    output = tool_rag(message, history, cur_art)
    if output:
        generator = chain.stream({"message": output['prompt'], "history": history_langchain_format})
    else:
        generator = chain.stream({"message": message, "history": history_langchain_format})

    response = ""
    for gen in generator:
        response += gen

    history[-1] = (message, response)
    return history


def tool_rag(question, history, cur_art):
    """RAG (Retrieval-Augmented Generation) 도구를 사용하는 함수"""
    output = {}
    tool_calls = llm_with_tools.invoke(question).tool_calls
    if not tool_calls:
        return None
    context = ""
    for tool_call in tool_calls:
        tool_output = call_tool_func(tool_call, question)
        context, tool_name = tool_output
        context += str(context).strip()
        tool_name = str(tool_name)
    output['tool_name'] = tool_name

    if tool_name == "qa_with_explain":
        prompt = f"""
## Role: 박물관 도슨트

## Instruction
- 박물관 도슨트로서 주어진 ## 작품 정보 ## 를 읽고 상대방에게 질문에 답변합니다.
- 작품에 대한 설명을 요구할 경우 전반적인 작품에 대한 정보를 알기 쉽게 풀어서 전달합니다.
- 질문이 구체적인 경우 작품의 정보를 참고해 간단하고 명료하게 대답합니다.
- history를 참고하여 답변합니다.
- 친절하고 상냥하게 답변합니다.
- 한국어로 답변합니다.
- {cur_art.get('유물명') or cur_art.get('작품명')}에 대한 정보를 바탕으로 답변해야 합니다.
- 탈 박물관의 작품일 경우, 탈의 특징, 용도, 문화적 의미 등을 설명에 포함합니다.

## 작품 정보
유물명/작품명: {cur_art.get('유물명') or cur_art.get('작품명')}
시대/제작년도: {cur_art.get('시대') or cur_art.get('제작년도')}
발굴 장소/재료: {cur_art.get('발굴 장소') or cur_art.get('재료', '정보 없음')}
규격: {cur_art.get('규격', '정보 없음')}
작품 설명: {cur_art.get('작품 설명') or cur_art.get('작품설명')}
간단 요약: {cur_art.get('간단 요약') or cur_art.get('간단요약', '정보 없음')}
어린이 전용: {cur_art.get('어린이 전용') or cur_art.get('어린아이', '정보 없음')}
박물관 출처: {'탈 박물관' if cur_art.get('source') == 'tal' else '복천 박물관'}

## 질문
{question}
"""
        output['prompt'] = prompt
        return output
    
    return None

def call_tool_func(tool_call, question):
    """특정 도구를 호출하는 함수"""
    tool_name = tool_call["name"].lower()
    if tool_name not in globals():
        print("Tool not found", tool_name)
        return None
    selected_tool = globals()[tool_name]
    print(tool_call)

    if "query" not in tool_call["args"]:
        tool_call["args"] = {"query": question}
        print("query is empty", tool_call)

    return selected_tool.invoke(tool_call["args"]), tool_name

# Gradio 인터페이스 스타일 정의
css = """
#chat_img img {width: 600px; height: 600px; align-items: center;}
#chat_img Column {align-items: center;}
"""
# 이미지 파일 경로 지정
IMAGE_PATH = "C:\비전공자 하삼동\chatbot(upstagever)\data\dosent pictures.webp" # 여기에 실제 이미지 파일 경로를 입력하세요

# Gradio 인터페이스 구성
with gr.Blocks(title="박물관 AI 가이드", css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>박물관 AI 가이드</h1>")
    
    with gr.Row():
        search_art_tb = gr.Textbox(label="검색", info="작품에 대한 정보를 입력해주세요.")
        museum_select = gr.Radio(["복천 박물관", "탈 박물관", "모두"], label="박물관 선택", value="모두")
        
        # 고정된 경로의 webp 이미지를 표시
        user_image = gr.Image(value=IMAGE_PATH, width=400, height=400)
    
    initial_list, initial_dict = get_all_artworks()
    search_dropdown = gr.Dropdown(choices=initial_list, value='', label="작품 목록", info="작품을 선택해주세요.", interactive=True, filterable=True)
    search_btn = gr.Button("검색")
    search_to_meta = gr.Textbox(value=initial_dict, visible=False)
    cur_search_art_tb = gr.Textbox(label="search_art_tb", visible=False)

    is_child = gr.Checkbox(label="어린이용 설명", value=False)

    with gr.Row() as art_info:
        with gr.Column(scale=1):
            art_image = gr.Image(value=None, label="작품 이미지", scale=1, height=400, width=400)
        with gr.Column(scale=2):
            art_description = gr.Textbox(label="작품 설명", lines=10)
            generate_audio_checkbox = gr.Checkbox(label="음성으로 듣기", value=False)
            audio_player = gr.Audio(label="음성", visible=False, autoplay=True)

    # ChatBot
    with gr.Row() as art_chatbot:
        with gr.Column(scale=1.2):
            gr.Markdown("<h2 style='text-align: center; margin-bottom: 1rem'>AI 도슨트</h2>")
            gr.Markdown("<p style='text-align: center; font-size: 14px; margin-bottom: 1rem'>작품에 대해 궁금한 것이 있나요 ? 👀 </p>")
            
            custom_art_chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox()
            audio = gr.Audio(type="filepath")
            clear = gr.ClearButton([msg, custom_art_chatbot])

            # 음성 입력 처리
            audio.change(handle_audio, [audio, custom_art_chatbot, cur_search_art_tb], [msg, custom_art_chatbot, cur_search_art_tb])
            msg.submit(user, [msg, custom_art_chatbot, cur_search_art_tb], [msg, custom_art_chatbot], queue=False).then(
                chat, [custom_art_chatbot, cur_search_art_tb], [custom_art_chatbot])
            clear.click(lambda: None, None, custom_art_chatbot, queue=False)

    search_art_tb.submit(search_art, [search_art_tb, museum_select], [search_dropdown, search_to_meta], queue=False)
    
    search_btn.click(
        fn=search_art, 
        inputs=[search_art_tb, museum_select],
        outputs=[search_dropdown, search_to_meta]
    )
    
    search_dropdown.change(
        fn=dropdown_change, 
        inputs=[search_dropdown, search_to_meta, is_child, generate_audio_checkbox], 
        outputs=[art_image, cur_search_art_tb, art_description, audio_player, audio_player]
    )

    is_child.change(
        fn=dropdown_change,
        inputs=[search_dropdown, search_to_meta, is_child, generate_audio_checkbox],
        outputs=[art_image, cur_search_art_tb, art_description, audio_player, audio_player]
    )
    def update_audio_visibility(generate_audio):
        return gr.update(visible=generate_audio)
    
    generate_audio_checkbox.change(
        fn=update_audio_visibility,
        inputs=[generate_audio_checkbox],
        outputs=[audio_player]
    ).then(
        fn=dropdown_change,
        inputs=[search_dropdown, search_to_meta, is_child, generate_audio_checkbox],
        outputs=[art_image, cur_search_art_tb, art_description, audio_player, audio_player]
    )


# Gradio 앱 실행
demo.launch(share=True)
