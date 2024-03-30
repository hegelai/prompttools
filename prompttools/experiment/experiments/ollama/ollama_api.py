import requests
import json
import nltk
import time
import sys
def fix_split_text(text):
    words = nltk.word_tokenize(text)
    return ' '.join(words)


import requests
import json
import sys

import asyncio
# import docker

# async def print_container_logs(container_id, follow=True):
#     client = docker.from_env()
#     container = client.containers.get(container_id)

#     async def log_consumer():
#         for log in container.logs(stdout=True, stderr=True, follow=follow, stream=True):
#             print(log.decode().strip())

#     await asyncio.gather(log_consumer())


# def get_container_id(container_name):
#     client = docker.from_env()
#      # List all containers, only the running ones by default
#     containers = client.containers.list() 

#     for container in containers:
#         print(container.name)
#         if container_name in container.name:
#             print(container.id)
#             return container.id

#     return None


class OllamaAPI:
    def __init__(self, base_url,container_name='ollama_container'):
        self.base_url = base_url
        # self.container_id = get_container_id(container_name=container_name)
    

    def pull(self, model_name: str):
        url = f"{self.base_url}/api/pull"
        response = requests.post(url, json={"name": model_name})
        response.raise_for_status()
        # check if the model was pulled successfully
        if response.status_code == 200:
            print(f"Successfully pulled model {model_name}")
            print(response.text)
            # redirect the ouput of the command in docker to the terminal

            return True
        else:
            print(response.text)
            return False


    def generate_with_stream(self, model_name:str, prompt, logs= False):
        # if logs:
        #     loop = asyncio.get_event_loop()
        #     loop.run_until_complete(print_container_logs(self.container_id))
        data = {"model": model_name, "prompt": prompt}
        url = f"{self.base_url}/api/generate"
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        final_res = ""
        for line in response.iter_lines():
            body = json.loads(line)
            new_words = body.get("response", "")

            if body.get("done", False):
                sys.stdout.write(f"\r{new_words}\n")
                sys.stdout.flush()
                break

            final_res += new_words
            sys.stdout.write(new_words)
            sys.stdout.flush()

        return final_res

    def generate(self,model_name:str,prompt:str,logs=False):


        responses = []
        data = {"model": model_name, "prompt": prompt}
        url = f"{self.base_url}/api/generate"
        response = requests.post(url, json=data,stream=True)
        response.raise_for_status()
        if response.status_code == 200:
            # response.json() throws an error, which I think
            for res in response.text.split('\n')[:-1]:
                json_res = json.loads(res)
                if json_res["done"] != True:
                    responses.append(json_res["response"]) 
            return ''.join(responses)
                        
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None

    def create(self, model_name: str, model_path:str,logs=False):
      
        data = {"name": model_name, "path": model_path}
        url = f"{self.base_url}/api/create"
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        if response.status_code == 200:
            print(f"Successfully created model {model_name}")
            print(response.text)
            return True
        else:
            print(response.text)
            return False
    
    def push(self, model_name: str, model_path:str,user_name:str,password:str,logs=False):
     
        data = {"name": model_name, "path": model_path, "user_name":user_name, "password":password}
        url = f"{self.base_url}/api/push"
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        if response.status_code == 200:
            print(f"Successfully pushed model {model_name}")
            print(response.text)
            return True
        else:   
            print(response.text)
            return False
        


    def copy(self,sources:str,destination:str,logs):
      

        data = {"sources": sources, "destination": destination}
        url = f"{self.base_url}/api/copy"
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        if response.status_code == 200:
            print(f"Successfully copied {sources} to {destination}")
            print(response.text)
            return True
        else:
            print(response.text)
            return False
        
        



    def tags(self,logs=False):
        
        url = f"{self.base_url}/api/tags"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        if response.status_code == 200:
            print(f"Successfully retrieved tags")
            print(response.text)
            return True
        else:
            print(response.text)
            return False
    
        

    def delete(self, model_name: str,logs=False):
       
        
        data = {"name": model_name}
        url = f"{self.base_url}/api/delete"
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        if response.status_code == 200:
            print(f"Successfully deleted model {model_name}")
            print(response.text)
            return True
        else:
            print(response.text)
            return False
        







if __name__ == "__main__":
    # Replace 'your_base_url_here' with the actual base URL of your API
    base_url = "http://localhost:8080"
    api = OllamaAPI(base_url)

    prompt="Why is the sky blue?"
    generated_result = api.generate('llama2',prompt,logs=True)
    if generated_result:
        print()
        print()
        print()
        print("Generated Result:", generated_result)
    
   