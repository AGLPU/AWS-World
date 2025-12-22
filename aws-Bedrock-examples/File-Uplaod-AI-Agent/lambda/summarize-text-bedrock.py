import json
import boto3
import base64
from datetime import datetime
from email import message_from_bytes
import botocore.config

def extract_text_from_multipart(data):
    msg = message_from_bytes(data)
    text_content = ''
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get('Content-Disposition'))
            if content_type == 'text/plain' and 'attachment' not in content_disposition:
                text_content += part.get_payload(decode=True).decode() + "\n"
                break
    else:
        if msg.get_content_type() == 'text/plain':
            text_content += msg.get_payload(decode=True).decode('utf-8') + "\n"
    return text_content.strip() if text_content else None

def generate_summary_from_bedrock(content: str) -> str:
    prompt_text = f"""Please explain the following text in 200 words:\n\n{content}"""

    # Correctly structured payload for Bedrock text generation
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": prompt_text}
                ]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 800,
            "temperature": 0.7,
            "topP": 0.95,
            "stopSequences": ["\n"]   # stop when a newline occurs
        }
    }

    try:
        bedrock = boto3.client(
            'bedrock-runtime',
            region_name="ap-south-1",
            config=botocore.config.Config(
                read_timeout=900,
                connect_timeout=900,
                retries={'max_attempts': 10}
            )
        )

        response = bedrock.invoke_model(
            modelId='arn:aws:bedrock:ap-south-1:781817960287:inference-profile/global.amazon.nova-2-lite-v1:0',
            body=json.dumps(body).encode('utf-8'),
            contentType='application/json',
            accept='application/json'
        )

        # Decode the response
        response_body = json.loads(response['body'].read())

        print(f"response_body: {response_body}")

        # Many Bedrock models return text under "results" list
        if "results" in response_body:
            summary =  response_body.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")
        else:
            # Some models use "outputText" directly
            summary =  response_body.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")

        #print(f"summary: {summary}")
        return summary.strip()

    except Exception as e:
        print(f"Error invoking Bedrock model: {e}")
        return "Error generating summary."

def save_code_to_s3_bucket(summary_text: str, bucket_name: str, object_key: str):
    s3 = boto3.client('s3')
    try:
        s3.put_object(Bucket=bucket_name, Key=object_key, Body=summary_text.encode('utf-8'))
        print(f"Summary saved to s3://{bucket_name}/{object_key}")
    except Exception as e:
        print(f"Error saving summary to S3: {e}")

def generate_presigned_url(bucket_name, object_key, expiry=360):
    s3 = boto3.client("s3")
    return s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket_name,
            "Key": object_key
        },
        ExpiresIn=expiry
    )

def lambda_handler(event, context):
    decoded_data = base64.b64decode(event['body'])
    text_content = extract_text_from_multipart(decoded_data)

    if not text_content:
        return {
            'statusCode': 400,
            'body': json.dumps('No text content found')
        }

    summary = generate_summary_from_bedrock(text_content)

    print(f"summary: {summary}")

    if summary:
       current_time = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
       bucket_name = 'ag-bedrock-practice'
       object_key = f"summaries/summary_{current_time}.txt"
       save_code_to_s3_bucket(summary, bucket_name, object_key)
       signed_url = generate_presigned_url(bucket_name, object_key)

    return {
         'statusCode': 200,
         "body": json.dumps({ "message": "Summary generated", "url": signed_url})
    }
