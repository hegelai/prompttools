FROM ubuntu:latest

RUN apt-get update

RUN apt-get install -y git
RUN apt-get install -y curl
WORKDIR /home
RUN (cd /home; git clone https://github.com/jmorganca/ollama.git)
RUN apt-get install -y wget
RUN wget https://golang.org/dl/go1.20.7.linux-amd64.tar.gz -O go.tar.gz
RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN tar -C /usr/local -xzf go.tar.gz
ENV PATH=$PATH:/usr/local/go/bin
RUN go version
RUN rm go.tar.gz
WORKDIR /home/ollama
RUN go build .
EXPOSE 11434
RUN ./ollama serve & ./ollama pull llama2
ENTRYPOINT ["/home/ollama/ollama"]
ENV OLLAMA_HOST 0.0.0.0
CMD ["serve"]
