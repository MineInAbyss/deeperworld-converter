docker run --rm -it -v `pwd`:/data -p 25565:25565 \
-e LEVEL_TYPE=flat -e 'GENERATOR_SETTINGS={"type":"minecraft:overworld","generator":{"type":"minecraft:flat","settings":{"biome":"minecraft:the_void","lakes":false,"features":false,"layers":[],"structures":{"structures":{}}}}}' \
-e MODE=spectator \
-e MEMORY=2G -e EULA=TRUE -e USE_AIKAR_FLAGS=true --name mc itzg/minecraft-server

-e TYPE=FABRIC 