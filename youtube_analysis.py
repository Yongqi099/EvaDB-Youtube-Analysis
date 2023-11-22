import os
import shutil
from typing import Dict, List

import pandas as pd
import evadb

from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi

from enum import Enum

from googleapiclient.discovery import build
import csv


MAX_CHUNK_SIZE = 1000
DEFAULT_VIDEO_LINK = "https://www.youtube.com/watch?v=0E_wXecn4DU&pp=ygUKZGFpbHkgZG9zZQ%3D%3D"
SECOND_DEFAULT_LINK = "https://www.youtube.com/watch?v=42m9WKQ0jC0&pp=ygUKZGFpbHkgZG9zZQ%3D%3D"
DEFAULT_PROMPT = "Summarize the video"
VIDEO_SENTIMENT_ANALYSIS = ("Classify the text as positive or negative. Think step-by-step, and provide reasoning why "
                            "the text is identified as positive or negative.")
COMMENT_SENTIMENT_QUESTION = "Is this comment positive or negative? Answer only with the word <Positive> or <Negative>"
YT_CONST = "https://www.youtube.com/watch?v="
SEPARATOR = "\n===========================================\n"
NUM_COMMENTS = 10

OPENAI_API_KEY = ""
os.environ["YOUTUBE_API_KEY"] = "AIzaSyDtZjRWBX4OYcVUzQ6WmjYBeCtt_GGM3Gk"

# file paths
TRANSCRIPT_PATH = os.path.join("evadb_data", "tmp")
SUMMARY_PATH = os.path.join("evadb_data", "tmp", "summary.csv")
TRANSCRIPT_DIR = os.path.join("evadb_data", "transcripts")
COMMENT_PATH = os.path.join("evadb_data", "comments")


class UserInteraction(Enum):
    WELCOME = ("\n🔮 Welcome to EvaDB! This app lets you ask questions on any YouTube video.\nYou will only need to "
               "supply a Youtube URL.\n")
    ADD_VID = "\nWould you like to add an additional Video? (enter 'yes' if so): "
    ADD_ANALYSIS = "\nWould you like to analyze another Video? (enter 'yes' if so): "
    QUESTION_PROMPT = (f"{SEPARATOR}🪄 Ask anything about the video!\nQuestion (enter 'exit' to exit) (Press enter "
                       f"for default response):")
    GENERATING = SEPARATOR + "⏳ Generating response (may take a while)..."
    ANSWER = SEPARATOR + "✅ Answer:"
    VIDEO_SENTIMENT = f"{SEPARATOR}Video Sentiment Analysis: \n"
    COMMENT_SENTIMENT = f"{SEPARATOR}Comment Sentiment Analysis: \n"
    PROGRAM_HALT = SEPARATOR

    def __str__(self):
        return str(self.value)


global video_links
video_links = {}


def recreate_video_links():
    # Iterate over all files in the transcript_dir directory
    for filename in os.listdir(TRANSCRIPT_DIR):
        # Get the file name without the extension
        youtube_id = os.path.splitext(filename)[0]

        # Prepend YT_CONST to youtube_id
        video_link = YT_CONST + youtube_id

        # Add to video_links
        add_to_video_links(video_link)


def start():
    while True:
        try:
            direction = int(input("Enter 0 to use former data, Enter 1 to start anew: "))
            if direction not in [0, 1]:
                print("Invalid input. Please enter either 0 or 1.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid integer.\n")

    if direction == 1:
        cleanup()
    else:  # direction == 0
        recreate_video_links()


def partition_transcript(raw_transcript: str):
    """Group video transcript elements when they are too large.

    Args:
        raw_transcript (str): downloaded video transcript as a raw string.

    Returns:
        List: a list of partitioned transcript
    """
    if len(raw_transcript) <= MAX_CHUNK_SIZE:
        return [{"text": raw_transcript}]

    k = 2
    while True:
        if (len(raw_transcript) / k) <= MAX_CHUNK_SIZE:
            break
        else:
            k += 1

    chunk_size = int(len(raw_transcript) / k)

    partitioned_transcript = [
        {"text": raw_transcript[i: i + chunk_size]}
        for i in range(0, len(raw_transcript), chunk_size)
    ]
    if len(partitioned_transcript[-1]["text"]) < 30:
        partitioned_transcript.pop()
    return partitioned_transcript


def partition_summary(prev_summary: str):
    """Summarize a summary if a summary is too large.

    Args:
        prev_summary (str): previous summary that is too large.

    Returns:
        List: a list of partitioned summary
    """
    k = 2
    while True:
        if (len(prev_summary) / k) <= MAX_CHUNK_SIZE:
            break
        else:
            k += 1
    chunk_size = int(len(prev_summary) / k)

    new_summary = [
        {"summary": prev_summary[i: i + chunk_size]}
        for i in range(0, len(prev_summary), chunk_size)
    ]
    if len(new_summary[-1]["summary"]) < 30:
        new_summary.pop()
    return new_summary


def group_transcript(transcript: dict):
    """Group video transcript elements when they are too short.

    Args:
        transcript (dict): downloaded video transcript as a dictionary.

    Returns:
        str: full transcript as a single string.
    """
    new_line = ""
    for line in transcript:
        new_line += " " + line["text"]

    return new_line


def download_youtube_video_transcript(video_id: str):
    """Downloads a YouTube video's transcript.

    Args:
        video_id (str): url of the target YouTube video.
    """
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    print("✅ Video transcript downloaded successfully.")
    return transcript


def parse_video_links():
    for youtube_id, video_link in video_links.items():
        # Check if a transcript file already exists
        if os.path.exists(os.path.join(TRANSCRIPT_DIR, f"{youtube_id}.txt")): continue
        transcript = download_youtube_video_transcript(youtube_id)

        # Group the list of transcripts into a single raw transcript.
        if transcript is not None: transcript = group_transcript(transcript)
        write_transcript_to_file(youtube_id=youtube_id, transcript=transcript)


def get_openai_key():
    # get OpenAI key if needed
    try:
        if OPENAI_API_KEY == "":
            api_key = os.environ["OPENAI_API_KEY"]
        else:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    except KeyError:
        api_key = str(input("🔑 Enter your OpenAI key: "))
        os.environ["OPENAI_API_KEY"] = api_key


def cleanup():
    """Removes any temporary file / directory created by EvaDB."""
    if os.path.exists("evadb_data"):
        print("\nCleaning")
        shutil.rmtree("evadb_data")
        print("Cleaning Completed")


def write_transcript_to_file(youtube_id, transcript):
    """Writes a transcript to a text file.

    Args:
        youtube_id (str): The YouTube video ID.
        transcript (str): The transcript to write.
    """
    # Create the transcript directory if it doesn't exist
    if not os.path.exists(TRANSCRIPT_DIR):
        os.makedirs(TRANSCRIPT_DIR)

    # Write the transcript file in the transcript directory
    path = os.path.join(TRANSCRIPT_DIR, f"{youtube_id}.txt")
    with open(path, "w") as file:
        file.write(transcript)


def list_videos():
    """Lists all available transcripts.

    Returns:
        transcripts (list): A list of available transcripts.
    """
    # Print every video name
    print()
    for i, title in enumerate(video_links.values()):
        print(f"{i}. {title}")

    # Prompt the user to select a video number that correspond with a key
    while True:
        # Ask the user for input
        video_number = input("Enter the video number to analyze: ")

        # Try to convert the input to an integer
        video_number = int(video_number)

        # Check if the number is in the valid range
        if 0 <= video_number < len(video_links):
            # If it is, break the loop
            break
        else:
            print(f"Please enter a number between 0 and {len(video_links) - 1}.")

    # Return the corresponding key
    return list(video_links.keys())[video_number]


def read_transcript(youtube_id):
    """Reads a transcript from a text file.

    Args:
        youtube_id (str): The YouTube video ID.

    Returns:
        transcript (str): The transcript.
    """
    with open(os.path.join(TRANSCRIPT_DIR, f"{youtube_id}.txt"), "r") as file:
        return file.read()


def get_top_comments(video_id, max_results=NUM_COMMENTS):
    youtube = build('youtube', 'v3', developerKey=os.environ["YOUTUBE_API_KEY"])

    # Retrieve comments for the video
    comments = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        order='relevance',
        textFormat='plainText',
        maxResults=max_results
    ).execute()
    # Extract top comments
    top_comments = [comment['snippet']['topLevelComment']['snippet']['textDisplay'] for comment in comments['items']]

    return top_comments


def store_comments_as_csv(video_id, top_comments):
    # Ensure the comments directory exists
    os.makedirs(COMMENT_PATH, exist_ok=True)

    file_name = os.path.join(COMMENT_PATH, f"{video_id}.csv")

    with open(file_name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['comment'])  # Header
        for comment in top_comments:
            writer.writerow([comment])

    return file_name


def generate_summary(cursor: evadb.EvaDBCursor, table_name: str):
    """Generate summary of a video transcript if it is too long (exceeds llm token limits)

    Args:
        cursor (EVADBCursor): evadb api cursor.
        :param table_name: table to query
    """
    transcript_list = cursor.table(table_name).select("text").df()["text"]
    if len(transcript_list) == 1:
        summary = transcript_list[0]
        df = pd.DataFrame([{"summary": summary}])
        df.to_csv(SUMMARY_PATH)

        cursor.drop_table("Summary", if_exists=True).execute()
        cursor.query(
            """CREATE TABLE IF NOT EXISTS Summary (summary TEXT(100));"""
        ).execute()
        cursor.load(SUMMARY_PATH, "Summary", "csv").execute()
        return

    generate_summary_rel = cursor.table(table_name).select(
        "ChatGPT('summarize the video in detail', text)"
    )
    responses = generate_summary_rel.df()["chatgpt.response"]

    summary = ""
    for r in responses:
        summary += f"{r} \n"
    df = pd.DataFrame([{"summary": summary}])
    df.to_csv(SUMMARY_PATH)

    need_to_summarize = len(summary) > MAX_CHUNK_SIZE
    while need_to_summarize:
        partitioned_summary = partition_summary(summary)

        df = pd.DataFrame([{"summary": partitioned_summary}])
        df.to_csv(SUMMARY_PATH)

        cursor.drop_table("Summary", if_exists=True).execute()
        cursor.query(
            """CREATE TABLE IF NOT EXISTS Summary (summary TEXT(100));"""
        ).execute()
        cursor.load(SUMMARY_PATH, "Summary", "csv").execute()

        generate_summary_rel = cursor.table("Summary").select(
            "ChatGPT('summarize in detail', summary)"
        )
        responses = generate_summary_rel.df()["chatgpt.response"]
        summary = " ".join(responses)

        # no further summarization is needed if the summary is short enough
        if len(summary) <= MAX_CHUNK_SIZE:
            need_to_summarize = False

    # load final summary to table
    cursor.drop_table("Summary", if_exists=True).execute()
    cursor.query(
        """CREATE TABLE IF NOT EXISTS Summary (summary TEXT(100));"""
    ).execute()
    cursor.load(SUMMARY_PATH, "Summary", "csv").execute()


def generate_response(cursor: evadb.EvaDBCursor, question: str, table_name: str) -> str:
    """Generates question response with llm.

    Args:
        cursor (EVADBCursor): evadb api cursor.
        question (str): question to ask to llm.
        table_name (str): table to query from.

    Returns
        str: response from llm.
    """
    if cursor.query(f"SELECT text FROM {table_name};").df().size == 1:
        response_df = cursor.query(f"SELECT ChatGPT('{question}', text) FROM {table_name};").df()
        if not response_df.empty:
            return response_df.iloc[0, 0]
        else:
            # Handle the case where the DataFrame is empty
            return "No response available"  # Or any other appropriate message
    else:
        # generate summary of the video if it's too long
        if not os.path.exists(SUMMARY_PATH):
            generate_summary(cursor, table_name)

        return (
            cursor.table("Summary")
            .select(f"ChatGPT('{question}', summary)")
            .df()["chatgpt.response"][0]
        )


def initial_response(cursor: evadb.EvaDBCursor, table_name: str, comment_table_name: str):
    print(UserInteraction.GENERATING)

    summary = generate_response(cursor, DEFAULT_PROMPT, table_name)
    sentiment = generate_response(cursor, VIDEO_SENTIMENT_ANALYSIS, table_name)
    comments_df = cursor.table(comment_table_name).select(f"ChatGPT('{COMMENT_SENTIMENT_QUESTION}', comment)").df()

    # Count sentiment occurrences
    counter = [0, 0, 0]
    for index, row in comments_df.iterrows():
        response = row.iloc[0]
        if 'positive' in response.lower():
            counter[0] += 1
        elif 'negative' in response.lower():
            counter[1] += 1
        else:
            counter[2] += 1

    print(UserInteraction.ANSWER)
    print(summary)
    print(UserInteraction.VIDEO_SENTIMENT)
    print(sentiment)
    print(UserInteraction.COMMENT_SENTIMENT)
    # Print sentiment counts
    print(f"{counter[0]} positive, {counter[1]} negative, {counter[2]} neutral/unsure")
    print(SEPARATOR)


def add_to_video_links(video_link: str):
    youtube = YouTube(video_link)
    youtube_id = youtube.video_id
    video_title = youtube.title
    video_links[youtube_id] = video_title


def receive_user_input():
    """
    Receives user input.
    """
    while True:
        # get Youtube video url
        video_link = str(input(
            "🌐 Enter the URL of the YouTube video (press Enter when done): "
        ))

        if video_link == "": video_link = DEFAULT_VIDEO_LINK

        # Check if the URL is a valid YouTube URL
        if video_link.startswith(YT_CONST):
            add_to_video_links(video_link)
            break

        else:
            print("⚠️ Please enter a valid YouTube URL.\n")


def video_title_to_table_name(youtube_id: str):
    """
    Convert youtube video title to acceptable table names
    :param youtube_id: youtube video id
    :return: table_name
    """
    # Replace spaces with underscores
    video_title = video_links.get(youtube_id).replace(' ', '_')
    # Remove special characters
    video_title = ''.join(ch for ch in video_title if ch.isalnum() or ch == '_')
    # Use the sanitized video title as the table name
    table_name = f"{video_title}_Transcript"
    return table_name


def partition_transcript_logic(transcript: str, youtube_id: str):
    # Partition the transcripts if they are too big to circumvent LLM token restrictions.
    path = ""
    if transcript is not None:
        partitioned_transcript = partition_transcript(transcript)
        df = pd.DataFrame(partitioned_transcript)
        # Name the CSV file based on the youtube_id and save it in the directory specified by TRANSCRIPT_PATH
        path = os.path.join(TRANSCRIPT_PATH, f"{youtube_id}.csv")
        df.to_csv(path)
    return path


def create_and_load_table(table_name: str, path: str):
    # Create a new table named based on the youtube_id
    cursor.drop_table(table_name=table_name, if_exists=True).execute()
    cursor.query(f"""CREATE TABLE IF NOT EXISTS {table_name} (text TEXT(50));""").execute()
    # Load the CSV file into the table
    cursor.load(path, table_name, "csv").execute()


def create_and_load_comment_table(table_name, path):
    # Drop the table if it already exists
    cursor.drop_table(table_name=table_name, if_exists=True).execute()
    # Create the table with the expected columns
    cursor.query(f"CREATE TABLE IF NOT EXISTS {table_name} (comment TEXT(100));").execute()
    # Load the CSV file into the table, skipping the header row
    cursor.load(path, table_name, "csv", skiprows=1).execute()


def query_video(youtube_id: str):
    transcript = read_transcript(youtube_id)
    table_name = video_title_to_table_name(youtube_id)
    path = partition_transcript_logic(transcript, youtube_id)
    create_and_load_table(table_name, path)

    # Retrieve top comments
    top_comments = get_top_comments(youtube_id)
    # Store comments as CSV
    comment_path = store_comments_as_csv(youtube_id, top_comments)
    comment_table_name = table_name[:-len('Transcript')].rstrip('_') + "_Comments"
    # Create and load the table with comments
    create_and_load_comment_table(comment_table_name, comment_path)

    while True:
        question = str(input(UserInteraction.QUESTION_PROMPT))
        if question.lower() == "exit":
            break

        if question == "":
            initial_response(cursor, table_name, comment_table_name)
            continue

        # Generate response with chatgpt udf
        print(UserInteraction.GENERATING)
        response = generate_response(cursor, question, table_name)
        print(UserInteraction.ANSWER)
        print(response)
        print(SEPARATOR)


if __name__ == "__main__":

    print(UserInteraction.WELCOME)

    try:
        start()

        # establish evadb api cursor
        cursor = evadb.connect().cursor()

        while True:
            receive_user_input()
            if str(input(UserInteraction.ADD_VID)).lower() not in ["y", "yes"]: break

        parse_video_links()
        get_openai_key()

        while True:
            choice = list_videos()
            query_video(choice)
            if str(input(UserInteraction.ADD_ANALYSIS)).lower() not in ["y", "yes"]: break

        print("✅ Session ended.")
        print(SEPARATOR)
    except Exception as e:
        print("❗️ Session ended with an error.")
        print(e)
