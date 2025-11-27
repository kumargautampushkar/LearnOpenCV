FROM osrf/ros:jazzy-desktop

# Install basic dev tools
RUN apt-get update && apt-get install -y \
    nano \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (matches standard Podman host UID 1000)
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up ROS environment for the user
USER $USERNAME
WORKDIR /home/$USERNAME
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc