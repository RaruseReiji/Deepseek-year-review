# Copyright (c) 2026 RaruseReiji
# Licensed under the MIT License.

import json
import matplotlib
matplotlib.use("Agg")
import os
import re
import time
import jieba
import requests
import threading
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from wordcloud import WordCloud
import platform
import sys
import traceback


# ================= 配置区域 =================
JSON_FILE_PATH = r'conversations.json'  # 你的导出的JSON文件路径
API_KEY = ""  # 可选择在此填入你的API Key，则跳过读取外部文件
API_URL = "https://api.deepseek.com/chat/completions"
# FONT_PATH = "C:\\Windows\\Fonts\\simhei.ttf"  # Windows系统默认黑体路径，用于词云

# 时间筛选范围 (UTC+8)
# 格式: "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM:SS"
# 如果想统计所有时间，可以把年份写得很夸张，比如 1970 到 2099
START_TIME_STR = "2025-01-01 00:00:00"
END_TIME_STR = "2025-12-31 23:59:59"

# 图片和报告的输出目录
OUTPUT_DIR = "AI_Annual_Report_2025"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

def load_api_key(path="apikey.txt"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 API Key 文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

if API_KEY == "" :
    API_KEY = load_api_key()

if API_KEY == "" :
    raise KeyError(f"未找到API Key，请检查外部文件")

def get_font_path():
    system = platform.system()
    if system == "Windows":
        return r"C:\Windows\Fonts\msyh.ttc"
    elif system == "Darwin":  # macOS
        return "/System/Library/Fonts/PingFang.ttc"
    else:  # Linux
        return "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"

FONT_PATH = get_font_path()

# 标签集
TAG_SET = {
    "信息与知识", "写作与表达", "学习与考试", "编程与技术", "创作与灵感", 
    "角色扮演与模拟", "决策与建议", "生活与实用", "娱乐与消遣", 
    "元对话与AI本身", "自然科学", "工程与技术", "医学与健康", 
    "社会科学", "艺术与人文", "商业与管理", "数学与统计"
}

# 忽略词（用于词云清洗）
STOP_WORDS = {
    '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', 
    '自己', '这', '那', '如何', '什么', '怎么', '这个', '那个', '因为', '所以','可能',
    '比如', '或者', '使用', '需要', '是否', '可以', '如果', '用户', 
    'DeepSeek', 'Assistant', 'User', 'System'
}
# ===========================================

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "font.size": 12
})

def global_exception_handler(exc_type, exc_value, exc_traceback):
    # KeyboardInterrupt 不记录，允许用户正常中断
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    log_file = "error.log"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Python: {platform.python_version()}\n")
        f.write("Exception:\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)

    print("\n程序发生错误，已生成 error.log 文件。")
    print("请将 error.log 发送给开发者以便排查问题。")


def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_time(time_str):
    if not time_str:
        return None
    # 处理可能存在的毫秒截断问题
    try:
        return datetime.fromisoformat(time_str)
    except:
        return None

def serialize_for_json(obj):
    """
    递归地将 datetime 对象转换为 ISO 字符串，供 JSON 序列化
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(i) for i in obj]
    else:
        return obj


def clean_text_for_wordcloud(text):
    # 移除Markdown符号, HTML标签, 网址等
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL) # 移除代码块
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[\#\*\-\>\[\]\(\)\`]', '', text) # 移除Markdown格式字符
    text = re.sub(r'http\S+', '', text) # 移除链接
    # 移除非中英文字符（保留基本标点供分词，分词后再过滤）
    return text

def analyze_part_one(data):
    print(">>> 开始第一部分：本地数据统计与可视化...")
    
    # 确保输出目录存在
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 解析筛选时间
    try:
        start_filter = datetime.fromisoformat(START_TIME_STR).replace(tzinfo=None) # 简化比较，转为naive或统一时区
        end_filter = datetime.fromisoformat(END_TIME_STR).replace(tzinfo=None)
    except ValueError:
        # 容错处理：如果格式不对，尝试补全
        start_filter = datetime.strptime(START_TIME_STR, "%Y-%m-%d %H:%M:%S")
        end_filter = datetime.strptime(END_TIME_STR, "%Y-%m-%d %H:%M:%S")

    stats = {
        "total_conversations": 0,
        "first_conversation": None,
        "longest_duration_chat": None, # (id, title, duration)
        "max_regenerate_req": {"count": 0, "chat_id": "", "title": ""}, # 用户让模型重答次数最多
        "max_reask_resp": {"count": 0, "chat_id": "", "title": ""}, # 用户针对回答追问次数最多
        "most_turns_chat": {"count": 0, "chat_id": "", "title": ""},
        "total_requests": 0,
        "total_responses": 0,
        "error_counts": {
            "busy": 0,
            "cant_answer": 0
        },
        "error_timestamps": {
            "busy": [],
            "cant_answer": []
        },
        "search_stats": {
            "total_count": 0,
            "max_results_event": {"count": 0, "chat_id": ""},
            "total_items": 0,
            "site_counts": Counter()
        },
        "file_stats": {
            "total_count": 0,
            "type_counts": Counter(),
            "chat_has_file_count": 0
        },
        "model_usage": Counter(),
        "all_request_text": [],
        "all_response_text": [],
        "user_top_words": [],
        "AI_top_words": []
    }

        # 预处理：筛选符合时间范围的对话
    valid_conversations = []
    for c in data:
        if "title" not in c: continue
        
        c_time = parse_time(c.get("inserted_at"))
        if not c_time: continue
        
        # 移除时区信息进行比较（假设配置和数据都是同个时区逻辑，或者自行处理时区转换）
        # DeepSeek 导出通常是 UTC+8，这里简化处理
        c_time_naive = c_time.replace(tzinfo=None)
        
        if start_filter <= c_time_naive <= end_filter:
            valid_conversations.append(c)

    stats["total_conversations"] = len(valid_conversations)
    valid_conversations.sort(key=lambda x: x.get("inserted_at", "") or "9999")
    
    if valid_conversations:
        stats["first_conversation"] = {
            "date": valid_conversations[0].get("inserted_at"),
            "title": valid_conversations[0].get("title")
        }

    max_duration = -1
    
    for chat in valid_conversations:
        chat_id = chat.get("id")
        title = chat.get("title", "No Title")
        mapping = chat.get("mapping", {})
        chat_has_search = False
        chat_has_file = False  # 本次对话是否包含文件
        
        # 1. 时间跨度
        start = parse_time(chat.get("inserted_at"))
        end = parse_time(chat.get("updated_at"))
        if start and end:
            duration = (end - start).total_seconds()
            if duration > max_duration:
                max_duration = duration
                stats["longest_duration_chat"] = {"id": chat_id, "title": title, "duration_hours": round(duration/3600, 2)}

        # 遍历 Mapping 分析
        req_count = 0
        resp_count = 0
        max_chain_len = 0
        
        # 构建节点父子关系查找最长链路
        nodes = mapping.keys()
        # 简单的深度计算（由于JSON结构是平铺的mapping，这里做简化处理：统计Request/Response对）
        
        for node_id, node_data in mapping.items():
            if not node_data.get("message"): continue
            
            message = node_data["message"]
            model = message.get("model")
            fragments = message.get("fragments", [])
            children = node_data.get("children", [])
            msg_inserted_at = parse_time(message.get("inserted_at"))

            # --- 文件统计 (Files 在 message 层级) ---
            files = message.get("files", [])
            if files:
                chat_has_file = True
                stats["file_stats"]["total_count"] += len(files)
                for f in files:
                    fname = f.get("file_name", "")
                    # 简单获取后缀名
                    ext = fname.split('.')[-1].lower() if '.' in fname else 'unknown'
                    stats["file_stats"]["type_counts"][ext] += 1
            
            # 确定消息类型 & 内容提取
            msg_type = "UNKNOWN"
            full_content_text_for_check = "" # 用于检查错误的完整文本（包含THINK）
            clean_response_text_for_wc = ""  # 用于词云的纯净文本（不包含THINK）
            request_text_for_wc = ""
            
            for frag in fragments:
                f_type = frag.get("type")
                content = frag.get("content", "")
                
                # 拼接完整文本用于错误检测
                full_content_text_for_check += content

                if f_type == "REQUEST":
                    msg_type = "REQUEST"
                    request_text_for_wc += content
                    if model: stats["model_usage"][model] += 1
                
                elif f_type == "RESPONSE":
                    msg_type = "RESPONSE"
                    clean_response_text_for_wc += content # 只收集 RESPONSE 正文，不含 THINK
                    
                elif f_type == "THINK":
                    # THINK 内容仅用于完整性检查，不计入 AI 词云
                    pass 

                elif f_type == "SEARCH":
                    chat_has_search = True
                    results = frag.get("results", [])
                    search_items_count = len(results)
                    stats["search_stats"]["total_count"] += 1
                    stats["search_stats"]["total_items"] += search_items_count
                    if search_items_count > stats["search_stats"]["max_results_event"]["count"]:
                        stats["search_stats"]["max_results_event"] = {"count": search_items_count, "chat_id": chat_id}
                    for res in results:
                        site = (res.get("site_name") or "").strip()
                        if site: stats["search_stats"]["site_counts"][site] += 1

            # --- 错误统计与时间点记录 ---
            if msg_type == "RESPONSE":
                # 检查 full_content_text_for_check
                if "服务器繁忙，请稍后再试" in full_content_text_for_check: 
                    stats["error_counts"]["busy"] += 1
                    if msg_inserted_at: stats["error_timestamps"]["busy"].append(msg_inserted_at)

                if "你好，这个问题我暂时无法回答" in full_content_text_for_check or \
                   "对不起，我还没有学会如何思考这类问题" in full_content_text_for_check:
                    stats["error_counts"]["cant_answer"] += 1
                    if msg_inserted_at: stats["error_timestamps"]["cant_answer"].append(msg_inserted_at)

            # --- 文本收集 (词云) ---
            if msg_type == "REQUEST":
                req_count += 1
                stats["all_request_text"].append(request_text_for_wc)
                if len(children) > 1:
                    if len(children) > stats["max_regenerate_req"]["count"]:
                        stats["max_regenerate_req"] = {"count": len(children), "chat_id": chat_id, "title": title}

            elif msg_type == "RESPONSE":
                resp_count += 1
                stats["all_response_text"].append(clean_response_text_for_wc) # 仅添加清洗后的RESPONSE
                if len(children) > 1:
                     if len(children) > stats["max_reask_resp"]["count"]:
                        stats["max_reask_resp"] = {"count": len(children), "chat_id": chat_id, "title": title}

        stats["total_requests"] += req_count
        stats["total_responses"] += resp_count
        
        if chat_has_file:
            stats["file_stats"]["chat_has_file_count"] += 1
        
        # 估算最长对话轮数 (Request + Response)
        # 严谨的算法应该是DFS找最深叶子节点，这里用 req_count 近似，
        # 或者 寻找mapping中 parent链条最长的
        
        current_chain_max = 0
        # 简单的链路长度计算：找到所有叶子节点，回溯到root
        leaf_nodes = [nid for nid, n in mapping.items() if not n.get("children")]
        for leaf in leaf_nodes:
            depth = 0
            curr = leaf
            while curr:
                depth += 1
                curr = mapping[curr].get("parent")
            if depth > current_chain_max:
                current_chain_max = depth
        
        if current_chain_max > stats["most_turns_chat"]["count"]:
            stats["most_turns_chat"] = {"count": current_chain_max, "chat_id": chat_id, "title": title}

        chat["_has_search"] = chat_has_search
        chat["_has_file"] = chat_has_file

    stats["most_turns_chat"]["count"] = (stats["most_turns_chat"]["count"] - 1) // 2 # 除去根节点，按一次提问一次回答算一轮
    stats["max_regenerate_req"]["count"] = stats["max_regenerate_req"]["count"] - 1 # 系统高估，正常回答有1，新增才是重试。
    stats["max_reask_resp"]["count"] = stats["max_reask_resp"]["count"] - 1 # 系统高估，正常提问有1，新增才是重试。

    # === 可视化生成 ===
    
    # 1. 搜索来源直方图
    site_counts = stats["search_stats"]["site_counts"].most_common(15)
    if site_counts:
        sites, counts = zip(*site_counts)
        plt.figure(figsize=(12, 6))
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        plt.bar(sites, counts, color='skyblue')
        plt.title('Top 15 搜索来源网站')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'search_sources_chart.png'))
        print("已保存: search_sources_chart.png")

    # 2. 词云生成函数
    def generate_wc(text_list, filename, title_text):
        full_text = " ".join(text_list)
        full_text = clean_text_for_wordcloud(full_text)
        
        # 结巴分词
        words = jieba.cut(full_text)
        # filtered_words = [w for w in words if len(w) > 1 and w not in STOP_WORDS and not w.isnumeric()]
        # 增加过滤逻辑：去除纯数字、纯符号、长度为1的词
        filtered_words = [
            w for w in words 
            if len(w) > 1 
            and w not in STOP_WORDS 
            and not w.isnumeric()
            and not re.match(r'^[^\u4e00-\u9fa5a-zA-Z0-9]+$', w) # 去除纯标点符号
        ]
        
        if not filtered_words:
            print(f"Warning: Not enough text for {filename}")
            return
        
        word_counter = Counter(filtered_words)
        top_words = word_counter.most_common(20)

        wc = WordCloud(
            font_path=FONT_PATH,
            width=2400, height=1400,
            scale=2,
            background_color='white',
            max_words=200,
            collocations=False
        ).generate(" ".join(filtered_words))
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title_text)
        save_path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(save_path)
        print(f"已保存: {filename}")
        
        return top_words

    stats["user_top_words"] = generate_wc(stats["all_request_text"], "wordcloud_user.png", "User 提问关键词")
    stats["AI_top_words"] = generate_wc(stats["all_response_text"], "wordcloud_ai.png", "AI 回答关键词")
    
    # 清理内存中的大文本列表，后续不需要
    del stats["all_request_text"]
    del stats["all_response_text"]
    
    serializable_stats = serialize_for_json(stats)

    with open(os.path.join(OUTPUT_DIR, "statistic_results.json"), "w", encoding="utf-8") as f:
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2)

    return stats

# ================= 第二部分：API 总结 =================

def call_deepseek_api(messages, model = "deepseek-chat", max_tokens = 8192):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 1.0,
        "max_tokens": max_tokens # 增加max_tokens以适应长总结生成
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=180) # 增加超时时间
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"API Error: {e}")
        return None

def process_single_chat_summary(chat):
    chat_id = chat.get("id")
    title = chat.get("title", "无标题")
    mapping = chat.get("mapping", {})
    has_search = chat.get("_has_search", False)
    has_file = chat.get("_has_file", False) # 获取文件标记
    
    # 构建对话文本，保留逻辑结构较为复杂，这里简化为线性提取
    # 提取所有 REQUEST 和 RESPONSE，按顺序拼接
    # 更严谨的做法是按 mapping 的 root -> children 遍历，这里做简化处理：
    # 将所有 message 按 inserted_at 排序
    
    msgs = []
    for mid, data in mapping.items():
        if data.get("message"):
            msgs.append(data["message"])
    
    msgs.sort(key=lambda x: x.get("inserted_at", "") or "")
    
    transcript = f"对话标题: {title}\n\n"
    for m in msgs:
        fragments = m.get("fragments", [])
        for f in fragments:
            if f['type'] == 'REQUEST':
                transcript += f"User: {f.get('content', '')}\n"
            elif f['type'] == 'RESPONSE':
                transcript += f"AI: {f.get('content', '')}\n"
    
    # 截断过长对话以节省Token
    if len(transcript) > 10000:
        transcript = transcript[:4000] + "\n...[中间省略]...\n" + transcript[-5000:]

    prompt_content = f"""
    请对以下用户与AI的对话进行简要总结（250字以内）。
    并从给定的Tags中选择最合适的（可多选）：{TAG_SET}
    
    输出格式必须为JSON格式，包含两个字段：
    "summary": "总结内容",
    "tags": ["Tag1", "Tag2"]
    
    对话内容：
    {transcript}
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant specialized in summarizing conversations."},
        {"role": "user", "content": prompt_content}
    ]
    
    result = call_deepseek_api(messages)
    
    tags = []
    summary = ""
    
    if result:
        # 尝试提取 JSON
        try:
            # 简单的 JSON 提取逻辑，防止 AI 包含 Markdown 代码块
            json_str = result
            if "```json" in result:
                json_str = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                json_str = result.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
            summary = data.get("summary", "")
            tags = data.get("tags", [])
        except:
            # Fallback: 如果 JSON 解析失败，简单正则提取
            summary = result
            tags = [t for t in TAG_SET if t in result]

    return {
        "id": chat_id,
        "title": title,
        "inserted_at": chat.get("inserted_at"),
        "updated_at": chat.get("updated_at"),
        "summary": summary,
        "tags": tags,
        "has_search": has_search,
        "has_file": has_file # 返回该标记
    }

def analyze_part_two(data, max_workers=20):
    print(f">>> 开始第二部分：API 调用总结 (并发数: {max_workers})...")
    conversations = [c for c in data if "title" in c]
    
    # 限制处理数量用于测试，如果需要跑全量请注释下面这行
    # conversations = conversations[:20] 
    
    results = []
    lock = threading.Lock()
    total = len(conversations)
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chat = {executor.submit(process_single_chat_summary, chat): chat for chat in conversations}
        
        for future in as_completed(future_to_chat):
            res = future.result()
            with lock:
                results.append(res)
                completed += 1
                if completed % 10 == 0:
                    print(f"进度: {completed}/{total}")
    
    # 保存中间结果
    with open(os.path.join(OUTPUT_DIR, "summary_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

# ================= 第三部分：生成最终报告 =================

def generate_final_report(stats, summary_results):
    print(">>> 开始第三部分：生成最终年度总结报告...")
    
    # 1. 整理 Tag 统计
    all_tags = []
    for item in summary_results:
        all_tags.extend(item['tags'])
    tag_counts = Counter(all_tags)
    
    # 找出出现最多的 Tag
    most_common_tag = tag_counts.most_common(1)[0] if tag_counts else ("无", 0)

    # # 绘制 Tag 频数直方图
    # if tag_counts:
    #     # 排序
    #     sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    #     tags_x, counts_y = zip(*sorted_tags)
        
    #     plt.figure(figsize=(14, 8))
    #     plt.rcParams['font.sans-serif'] = ['SimHei']
    #     plt.rcParams['axes.unicode_minus'] = False
    #     plt.bar(tags_x, counts_y, color='lightgreen')
    #     plt.title('对话主题标签分布')
    #     plt.xlabel('标签')
    #     plt.ylabel('频次')
    #     plt.xticks(rotation=45, ha='right')
    #     plt.tight_layout()
    #     plt.savefig('tag_distribution.png')
    #     print("已保存: tag_distribution.png")

    # 整理「搜索 × Tag」

    search_tag_counter = Counter()

    for item in summary_results:
        if item.get("has_search"):
            for t in item["tags"]:
                search_tag_counter[t] += 1
    
    sorted_search_tags = search_tag_counter.most_common() if search_tag_counter else []

    if sorted_search_tags:
        tags, counts = zip(*sorted_search_tags)
    else:
        tags, counts = [], []



    # 2. 准备给 AI 的数据
    # 将所有总结拼接起来，让 AI 阅读所有内容

    summary_results.sort(
        key=lambda x: x.get("inserted_at") or ""
    )

    all_summaries_text = "\n".join([
        f"- [{s['title']}] {s['summary']} (Tags: {', '.join(s['tags'])})"
        for s in summary_results
    ])

    AI_top_words_text = ", ".join(
        [f"{w}({c})" for w, c in stats["AI_top_words"][:20]]
    )

    user_top_words_text = ", ".join(
        [f"{w}({c})" for w, c in stats["user_top_words"][:20]]
    )

    prompt_stats = f"""
    【年度基础数据】
    - 总对话数: {stats['total_conversations']}
    - 第一条对话: {stats['first_conversation']}
    - 最长时长对话: {stats['longest_duration_chat']}
    - 用户重试最多次: {stats['max_regenerate_req']}
    - 用户追问最多次: {stats['max_reask_resp']}
    - 最长轮数对话: {stats['most_turns_chat']}
    - 总提问/回答数: {stats['total_requests']} / {stats['total_responses']}
    - 错误出现次数: {stats['error_counts']}
    - 联网搜索总次数: {stats['search_stats']['total_count']}
    - 最常搜索的Tag是{tags}，搜索了{counts}次
    - 模型使用偏好: {dict(stats['model_usage'])}
    
    【内容偏好数据 (Tags)】
    - 最常出现的主题: {most_common_tag[0]} (出现 {most_common_tag[1]} 次)
    - 详细Tag分布: {dict(tag_counts)}
    
    【用户提问高频词 Top20】
    {user_top_words_text}

    【AI回答高频词 Top20】
    {AI_top_words_text}
    
    【所有对话详细记录】
    以下是该用户今年所有对话的简要总结，已按时间顺序排列（从年初到年末），请仔细阅读这些内容，挖掘用户的性格、工作性质、兴趣爱好，并可关注用户关注点、问题复杂度和主题分布随时间的变化轨迹：
    {all_summaries_text}
    """
    
    final_prompt = f"""
    你是一位专业的数据分析师和文字工作者。请根据用户提供的“AI年度使用数据”和“所有对话记录”，写一份深情、幽默且具有深度洞察力的《2025 AI年度总结报告》。
    
    ### 写作目标
    这不是流水账，而是一份：
    - 有洞察力的年度画像
    - 能“读懂这个用户”的深度总结
    - 读完后让用户感到“被理解、被记录、被陪伴”
    
    要求：
    1. 标题自拟，要吸引人。
    2. 全文通过阅读【所有对话详细记录】来构建精准的“用户画像”。
    3. 结合数据进行解读。例如：如果搜索次数多，说明用户喜欢考证；如果Tag集中在“编程”，说明是技术人员。
    4. 分章节叙述，例如“最关心的领域”、“深夜的思考”、“成长的足迹”等。
    5. 引用具体的对话案例（从提供的总结记录中挑选）来佐证你的观点。
    6. 结尾要升华主题，展望未来。
    7. 必须使用 Markdown 格式，允许使用二级、三级标题。
    
    数据如下：
    {prompt_stats}
    """
    
    messages = [
        {"role": "user", "content": final_prompt}
    ]
    
    print(">>> 正在调用大模型API生成年度报告，用时较长，预计2分钟，请耐心等待……")

    report_content = call_deepseek_api(messages, "deepseek-reasoner", 12800)
    
    if report_content:
        with open(os.path.join(OUTPUT_DIR, "2025_AI_Annual_Report.md"), "w", encoding="utf-8") as f:
            f.write(report_content)
        print(">>> 报告生成完毕！请查看 2025_AI_Annual_Report.md")
    else:
        print(">>> 报告生成失败。")
    
    return report_content


def build_report(
    stats,
    summary_results,
    ai_summary_text="",
    figures_dir=FIGURES_DIR
):
    
    # 确保目录存在
    os.makedirs(figures_dir, exist_ok=True)
    report_file_path = os.path.join(OUTPUT_DIR, "report.md") # 报告也生成在指定文件夹下

    # ===== 时间排序 =====
    summary_results = sorted(
        summary_results,
        key=lambda x: x.get("inserted_at") or ""
    )

    # ===== Tag 统计 =====
    tag_counter = Counter()
    search_tag_counter = Counter()
    file_tag_counter = Counter() # 新增：文件 x Tag

    for item in summary_results:
        for t in item["tags"]:
            tag_counter[t] += 1
            if item.get("has_search"):
                search_tag_counter[t] += 1
            if item.get("has_file"):
                file_tag_counter[t] += 1

    # 1. Search x Tag 图表
    if search_tag_counter:
        # 按次数从高到低排序
        sorted_search_tags = search_tag_counter.most_common()
        tags, counts = zip(*sorted_search_tags)
        plt.figure(figsize=(14, 8))
        plt.bar(tags, counts)
        plt.title('联网搜索 × 对话主题 Tag 分布')
        plt.xlabel('Tag')
        plt.ylabel('搜索次数')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'search_tag_distribution.png'))
        print("已保存: search_tag_distribution.png")
    else:
        print("未检测到任何带搜索的对话，跳过 搜索 × Tag 图表生成")

    # 2. File x Tag 图表 (新增)
    if file_tag_counter:
        sorted_file_tags = file_tag_counter.most_common()
        tags, counts = zip(*sorted_file_tags)
        plt.figure(figsize=(14, 8))
        plt.bar(tags, counts, color="orange")
        plt.title('文件上传 × 对话主题 Tag 分布')
        plt.xlabel('Tag')
        plt.ylabel('上传文件对话次数')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'file_tag_distribution.png'))
        plt.close()
        print("已保存: file_tag_distribution.png")

    # 3. 文件类型饼图 (新增)
    file_types = stats["file_stats"]["type_counts"]
    if file_types:
        # 只取 Top 10，其他的归为 Other
        top_files = file_types.most_common(10)
        labels, counts = zip(*top_files)
        other_count = sum(file_types.values()) - sum(counts)
        if other_count > 0:
            labels = labels + ('Other',)
            counts = counts + (other_count,)
            
        plt.figure(figsize=(8, 8))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title(f"上传文件类型分布 (Total: {stats['file_stats']['total_count']})")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'file_type_pie.png'))
        plt.close()
        print("已保存: file_type_pie.png")

    # 4. 繁忙/报错时间轴分布 (新增)
    busy_times = stats.get("error_timestamps", {}).get("busy", [])
    cant_answer_times = stats.get("error_timestamps", {}).get("cant_answer", [])
    
        # 定义一个内部绘图函数，避免重复代码
    def plot_error_bar(timestamps, color, title, filename):
        if not timestamps:
            return
        
        # 1. 数据聚合：将时间戳转为日期，并统计每天的次数
        date_counts = Counter([t.date() for t in timestamps])
        sorted_dates = sorted(date_counts.keys())
        counts = [date_counts[d] for d in sorted_dates]
        
        # 2. 绘图
        plt.figure(figsize=(15, 4)) # 高度稍微增加一点，方便看柱子
        # width=0.6 让柱子之间有点间隙
        plt.bar(sorted_dates, counts, color=color, alpha=0.7, width=0.6)
        
        # 3. 格式化 X 轴 (日期)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # 自动调整刻度密度，避免日期重叠
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # 4. 格式化 Y 轴 (只显示整数刻度)
        max_count = max(counts)
        if max_count < 10:
            plt.yticks(range(0, max_count + 2)) # 数量少时强制显示整数刻度
        
        plt.grid(axis='y', linestyle='--', alpha=0.3) # 加横向网格线辅助读数
        plt.title(f'{title} (总计: {sum(counts)}次)')
        plt.xticks(rotation=0) # 日期横着放，如果太挤会自动调整
        plt.tight_layout()
        
        save_path = os.path.join(figures_dir, filename)
        plt.savefig(save_path)
        plt.close()
    
    # 4.1 Busy 图 (红色条形图)
    plot_error_bar(busy_times, 'red', '服务器繁忙 (Busy) 每日频次', 'error_busy_timeline.png')
    
    # 4.2 Cant Answer 图 (蓝色条形图)
    plot_error_bar(cant_answer_times, 'blue', '无法回答 (Refusal) 每日频次', 'error_cant_answer_timeline.png')

    # 5. Tag 分布图
    most_common_tag = tag_counter.most_common(1)[0] if tag_counter else ("无", 0)
    # 移动位置，统一管理
    if tag_counter:
        sorted_tags = sorted(tag_counter.items(), key=lambda x: x[1], reverse=True)
        tags_x, counts_y = zip(*sorted_tags)
        plt.figure(figsize=(14, 8))
        plt.bar(tags_x, counts_y, color='lightgreen')
        plt.title('对话主题标签分布')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'tag_distribution.png'))
        plt.close()


    # ===== Markdown 正文 =====
    rel_fig_dir = "figures"

    lines = []

    lines.append(f"# {START_TIME_STR[:4]} · Deepseek 对话总结报告\n")
    lines.append(f"统计范围：{START_TIME_STR} 至 {END_TIME_STR}\n")

    lines.append("## 一、年度概览\n")
    lines.append(f"- **总对话数**：{stats['total_conversations']}")
    lines.append(f"- **总提问 / 回答数**：{stats['total_requests']} / {stats['total_responses']}")
    lines.append(f"- **深度思考 / 普通模式** 次数：{stats['model_usage']['deepseek-reasoner']} 次 /{stats['model_usage']['deepseek-chat']} 次")
    lines.append(f"- **联网搜索次数**：{stats['search_stats']['total_count']}")
    lines.append(f"- **上传文件总数**：{stats['file_stats']['total_count']}")
    lines.append(f"- **时间最长的对话**： \"{stats['longest_duration_chat']['title']}\" ，持续了{stats['longest_duration_chat']['duration_hours']}小时")
    lines.append(f"- **交流轮数最多的对话**： \"{stats['most_turns_chat']['title']}\" ，来回了{stats['most_turns_chat']['count']}轮")
    lines.append(f"- **重新提问最多的对话**： \"{stats['max_reask_resp']['title']}\" ，重新问了{stats['max_reask_resp']['count']}次")
    lines.append(f"- **要求D老师重新回答最多的对话**： \"{stats['max_regenerate_req']['title']}\" ，重新回答了{stats['max_regenerate_req']['count']}次")

    lines.append(f"- **最常出现主题**：{most_common_tag[0]}（{most_common_tag[1]} 次）\n")
    busy_cnt = stats['error_counts']['busy']
    cant_cnt = stats['error_counts']['cant_answer']
    lines.append(f"- **\"服务器繁忙，请稍后再试\"**：{busy_cnt} 次，D老师真的很忙啊！\n")
    lines.append(f"- **\"你好，这个问题我暂时无法回答\"**：{cant_cnt} 次，所以你平时在问些啥？\n\n")
    if busy_times:
        lines.append(f"![]({rel_fig_dir}/error_busy_timeline.png)\n")
    
    if cant_answer_times:
        lines.append(f"![]({rel_fig_dir}/error_cant_answer_timeline.png)\n")
        
    if not busy_times and not cant_answer_times:
         lines.append("今年非常顺利，没有遇到过错误。\n")

    lines.append("## 二、使用行为与偏好分析\n")

    lines.append("### 1. 提问与回答关键词\n")
    lines.append(f"![用户提问词云]({rel_fig_dir}/wordcloud_user.png)")
    lines.append(f"![AI 回答词云]({rel_fig_dir}/wordcloud_ai.png)\n")

    lines.append("### 2. 对话主题分布\n")
    lines.append(f"![Tag 分布]({rel_fig_dir}/tag_distribution.png)\n")

    lines.append("### 3. 联网搜索行为\n")
    lines.append(f"![搜索来源分布]({rel_fig_dir}/search_sources_chart.png)")
    lines.append(f"![搜索 × Tag 分布]({rel_fig_dir}/search_tag_distribution.png)\n")

    lines.append("### 4.文件投喂\n")
    if stats['file_stats']['total_count'] > 0:
        lines.append(f"![]({rel_fig_dir}/file_type_pie.png)\n")
        lines.append(f"![]({rel_fig_dir}/file_tag_distribution.png)\n")
    else:
        lines.append("今年没有上传过文件。\n")

    if ai_summary_text:
        lines.append("## 三、AI 生成的年度深度总结\n")
        lines.append(ai_summary_text.strip())
        lines.append("\n---\n")
        lines.append("## 四、全年对话时间线摘要\n")
    else:
        lines.append("## 三、全年对话时间线摘要\n")

    for item in summary_results:
        date = item.get("inserted_at", "")[:10]
        tags = ", ".join(item["tags"])
        lines.append(f"### {date} · {item['title']}")
        lines.append(f"- **Tags**：{tags}")
        lines.append(f"- **是否联网搜索**：{'是' if item.get('has_search') else '否'}")
        lines.append(f"- **是否上传文件**：{'是' if item.get('has_file') else '否'}")
        lines.append(f"- **摘要**：{item['summary']}\n")

    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"最终报告已生成: {report_file_path}")
    
def downgrade_headers(text, level=2):
    """
    将 ATX 风格 Markdown 标题整体降级（# 数量增加）
    """
    def repl(match):
        hashes = match.group(1)
        title = match.group(2)
        return "#" * (len(hashes) + level) + " " + title

    pattern = re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)
    return pattern.sub(repl, text)

# ================= 主程序 =================

sys.excepthook = global_exception_handler

if __name__ == "__main__":
    if not os.path.exists(JSON_FILE_PATH):
        print(f"错误：未找到文件 {JSON_FILE_PATH}")
    else:
        # 1. 加载数据
        raw_data = load_data(JSON_FILE_PATH)
        
        # 2. 本地统计 & 绘图
        statistics = analyze_part_one(raw_data)
        
        # 3. API 批量总结 (如果不希望消耗Token，可以注释掉下面两行，手动伪造 summary_results)
        # 注意：全量跑可能消耗较多Token和时间
        summaries = analyze_part_two(raw_data, max_workers=20)
        
        # 4. 最终报告
        ai_final_summary = generate_final_report(statistics, summaries) or ""

        ai_final_summary_2 = downgrade_headers(ai_final_summary)

        # 5. 生成文件
        build_report(statistics, summaries, ai_final_summary_2)
        input("按Enter键继续……")