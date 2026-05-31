FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

RUN rm -f /etc/apt/sources.list.d/yarn.list /usr/share/keyrings/yarn-archive-keyring.gpg \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        libgomp1 \
        curl \
        ca-certificates \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Install just
RUN pipx install rust-just

# Install uv
RUN pipx install uv
ENV UV_HTTP_TIMEOUT=300
ENV UV_LINK_MODE=copy

# AWS CLI
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
        unzip awscliv2.zip && ./aws/install && rm -rf aws/ awscliv2.zip; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
        curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" && \
        unzip awscliv2.zip && ./aws/install && rm -rf aws/ awscliv2.zip; \
    fi
