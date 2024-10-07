# Use the official Python 3.10 image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Lean 4
RUN wget -q https://raw.githubusercontent.com/leanprover-community/mathlib4/master/scripts/install_debian.sh \
    && bash install_debian.sh \
    && rm -f install_debian.sh \
    && source ~/.profile

# Add lean to the system's PATH
ENV PATH="/root/.elan/bin:$PATH"

# Install Python dependencies (assumes that the requirements.txt is in the repository)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Build Mathlib4 during the image build
WORKDIR /app/mathlib4
RUN lake build

# Set default work directory back to project root
WORKDIR /app

# Expose port if needed (optional)
# EXPOSE 8080

CMD ["bash"]
