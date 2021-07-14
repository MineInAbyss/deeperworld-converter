SET data=%~dp0
echo "data dir %data%"

docker run --rm -v %data%:/data -p 25565:25565 -e MEMORY=2G -e EULA=TRUE -e USE_AIKAR_FLAGS=true --name mc itzg/minecraft-server