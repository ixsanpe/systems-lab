# Apertium spa<->arg in a container: the most reliable route on macOS, where the
# language pair isn't packaged for Homebrew. Debian ships apertium-spa-arg 0.6.0.
#
#   docker build -t argmt-apertium -f docker/apertium.Dockerfile .
#   echo "Hola, mundo." | docker run -i --rm argmt-apertium spa-arg
FROM debian:trixie-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends apertium apertium-spa-arg \
    && rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["apertium", "-u"]
CMD ["spa-arg"]
