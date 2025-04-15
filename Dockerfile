# AgentK Dockerfile
#
# This Dockerfile builds a container for running the AgentK application.
# It includes Python, Rust, and system dependencies needed for the application.
#
# Base image: Python 3 on Debian Bullseye
FROM python:3-bullseye

# Update package lists and install Rust (required for some Python dependencies)
RUN apt-get -y update
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH /root/.cargo/bin:$PATH

# Install system dependencies from the apt-packages-list.txt file
WORKDIR /tmp
COPY apt-packages-list.txt /tmp/apt-packages-list.txt
# Fix Windows line endings if present
RUN sed -i 's/\r$//' apt-packages-list.txt
RUN xargs -a apt-packages-list.txt apt-get install -y

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r requirements.txt

# Set up application directory and copy code
WORKDIR /app
COPY . /app

# Default command: run the agent kernel
ENTRYPOINT ["python", "agent_kernel.py"]