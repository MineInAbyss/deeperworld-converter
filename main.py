import itertools
from amulet.api.selection import SelectionGroup, SelectionBox
from amulet.api.level import ImmutableStructure, World
from amulet.api.level.base_level.clone import clone
from amulet.api.data_types import Dimension, BlockCoordinatesAny, BlockCoordinates
import yaml
import amulet
from amulet.api.errors import ChunkLoadError, ChunkDoesNotExist
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from dataclasses import dataclass
import os
from typing import List
worlds_input_path = "/mnt/world/input/"
worlds_output_path = "/mnt/world/output/"
deeperworld_config_path = os.path.join(
    worlds_input_path, "plugins/DeeperWorld/config.yml")
converter_config_path = 'config.yml'


@dataclass
class ConverterConfig:
    world: str = "world"
    spacing: int = 16384
    overlap: int = 32
    low: int = -512
    high: int = 511
    # how high is the top of the first region? goes down with the list
    start_height: int = 256


@dataclass
class Vec3I:
    x: int = 0
    y: int = 0
    z: int = 0

    def cords(self):
        return self.x, self.y, self.z

    @staticmethod
    def zip(a, b, f):
        return Vec3I(f(a.x, b.x), f(a.y, b.y), f(a.z, b.z))

    @staticmethod
    def add(a, b):
        return Vec3I.zip(a, b, lambda a, b: a+b)

    @staticmethod
    def sub(a, b):
        return Vec3I.zip(a, b, lambda a, b: a-b)

    @staticmethod
    def minMax(a, b):
        return (
            Vec3I.zip(a, b, min),
            Vec3I.zip(a, b, max),
        )


@dataclass
class LayerConfig:
    name: str
    max: Vec3I
    min: Vec3I
    # where is min?
    pos: Vec3I
    world: str

    @property
    def size(self):
        return Vec3I.sub(self.max, self.min)

    @property
    def src_selection(self):
        return SelectionBox(self.max.cords(), self.min.cords())
    @property
    def offset(self):
        return Vec3I.sub(self.pos, self.max)
    @property
    def dst_selection(self):
        return SelectionBox(self.pos.cords(), Vec3I.add(self.pos, Vec3I.sub(self.min, self.max)).cords())


@dataclass
class DeeperWorldConfig:
    sections: List[LayerConfig]


def load_deeperworld_confg(config: ConverterConfig):
    with open(deeperworld_config_path, 'r') as f:
        data = (yaml.safe_load(f))
        sectionsIn = data["sections"]
        sectionsOut = []
        ref_top_pos = Vec3I()  # where is the
        for sectionIn in sectionsIn:
            name = sectionIn["name"]
            world = sectionIn["world"]
            ref_top = Vec3I(*sectionIn["refTop"])
            ref_bottom = Vec3I(*sectionIn["refBottom"])

            [x1, z1, x2, z2] = sectionIn["region"]
            region_min, region_max = Vec3I.minMax(
                Vec3I(x1, 0, z1), Vec3I(x2, 255, z2))

            region_min_pos = Vec3I.add(
                ref_top_pos, Vec3I.sub(region_min, ref_top))
            size = Vec3I.sub(region_max, region_min)
            ref_top_pos = Vec3I.add(
                ref_top_pos, Vec3I.sub(ref_bottom, ref_top))
            lc = LayerConfig(name, region_min, region_max,
                             region_min_pos, world)
            sectionsOut.append(lc)
        return sectionsOut


def load_converter_confg():
    with open(converter_config_path, 'r') as f:
        data = (yaml.safe_load(f))
        confg = ConverterConfig(**data)
        return confg


platform_version = ("java", (1, 17, 1))
dimension = "minecraft:overworld"


def mark_box_dirty(world: World, dimension, box):
    for cx, cz in box.chunk_locations():  # Mark the edits as dirty
        if(world.has_chunk(cx, cz, dimension)):
            chunk = world.get_chunk(cx, cz, dimension)
            chunk.changed = True


def progress_iter(gen, name):
    from progress.bar import Bar
    parts = 1e6
    with Bar(
            name,
            max=parts,
        suffix='%(percent).1f%% - %(eta)ds'
    ) as bar:
        last = 0
        try:
            while True:
                new = next(gen)
                if not isinstance(new, float):
                    (n, d) = new
                    new = n/d
                diff = int(parts*new) - int(parts*last)
                last = new
                if(diff >= 1):
                    bar.next(diff)
        except StopIteration as e:
            return e.value


def copy_region(src_world: World, src_region: SelectionBox, dst_world: World, dst_region: SelectionBox):
    # structure = progress_iter(ImmutableStructure.from_level_iter(
    #     src_world, SelectionGroup(src_region), dimension), "load")
    volume = src_region.volume
    cx = ((dst_region.max_x + dst_region.min_x) >> 1)
    cy = ((dst_region.max_y + dst_region.min_y) >> 1)  # Paste is from-centre
    cz = ((dst_region.max_z + dst_region.min_z) >> 1)
    dst_region_midpoint = (cx, cy, cz)
    print("clone", SelectionGroup(src_region), dst_region_midpoint)

    clone_op = dst_world.paste_iter(src_world, dimension, SelectionGroup(src_region),
    dimension, dst_region_midpoint)
    progress_iter(clone_op, "clone")
    #print("marking as dirty")
    mark_box_dirty(dst_world, dimension, dst_region)
    #print("did a copy")
    # for cx, cz in src_region.chunk_locations():
    #     print("copy chunk", cx, cz, "from", src_world.level_path)


def do_conversion(regions: List[LayerConfig]):
    level_out = amulet.load_level(os.path.join(worlds_output_path, "world"))
    #regions.reverse()

    def do(region):
        source_level = amulet.load_level(
            os.path.join(worlds_input_path, region.world))
        count = itertools.count()

        def sub_do(s):
            i = next(count)
            
            _, src_selection = s
            dst_selection= src_selection.transform((1,1,1), (0,0,0), region.offset.cords())
            #_, dst_selection = d
            print("region", i, src_selection, dst_selection)
            copy_region(source_level, src_selection,
                        level_out, dst_selection)

            if(i % 100 == 0 or True):
                progress_iter(level_out.save_iter(), "save")
                source_level.unload()
                level_out.unload()
        sub_chunk_size = 512/2
        print("layer", region.name, region.src_selection.chunk_count(sub_chunk_size), region.src_selection, region.dst_selection)

        list(map(sub_do, region.src_selection.chunk_boxes(sub_chunk_size)))
        progress_iter(level_out.save_iter(), "save")
        source_level.unload()
        level_out.unload_unchanged()

    # with ProcessPoolExecutor() as executor:
    for region in regions:
        do(region)
        # executor.
    # for region in regions:
    #     do(region)
    #     source_level=amulet.load_level(
    #         os.path.join(worlds_input_path, region.world))
    #     copy_region(source_level, region.src_selection,
    #                 level_out, region.dst_selection)
    #     source_level.close()

    level_out.close()
    # can't create anvil worlds becuse fml
    # (platform, version) = platform_version
    # level_out = amulet.level.formats.anvil_world.AnvilFormat(level_out_path)
    # level_out.create_and_open(platform, version)
    # level_out.save()
    # level_out.close()

    # (block, block_entity) = level.get_version_block(
    #     x, y, z, dimension, platform_version)
    # level_out.set_version_block(
    #     x, y, z, dimension, platform_version, block, block_entity)


converter_confg = load_converter_confg()
regions = load_deeperworld_confg(converter_confg)
do_conversion(regions)
