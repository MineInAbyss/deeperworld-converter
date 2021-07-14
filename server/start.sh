SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`

DATA=$SCRIPTPATH
echo "DATA=$DATA"
docker run --rm -v $DATA:/data -p 25565:25565 -e MEMORY=2G -e EULA=TRUE -e USE_AIKAR_FLAGS=true --name mc itzg/minecraft-server