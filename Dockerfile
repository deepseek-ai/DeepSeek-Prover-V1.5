FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Set environment variables
WORKDIR /app

# Install required system packages and Python
RUN apt-get update -y && apt-get install -y \
    curl \
    git
# Custom Lean 4 installation script without Visual Studio Code
RUN curl -y https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Copy the requirements.txt file
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Build Mathlib4 during the image build
# WORKDIR /app/mathlib4
# RUN lake exe cache get && lake build

# Set default work directory back to project root
WORKDIR /app

# Expose port if needed (optional)
# EXPOSE 8080

CMD ["bash"]

