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
    min_y: int = -64
    height: int = 384
    sub_chunk_size: int = 512
    @property
    def max_y(self):
        return self.max_y - self.height
    # how high is the top ref of the first region? goes down with the list
    start_height: int = 0


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
    print("copy", src_region, dst_region)

    if False:
        def do_single_copy(sp, dp):
            # print(sp, dp)
            (sx, sy, sz) = sp
            (dx, dy, dz) = dp
            (block, block_entity) = src_world.get_version_block(
                sx, sy, sz, dimension, platform_version)
            dst_world.set_version_block(
                dx, dy, dz, dimension, platform_version, block, block_entity)
        list(map(do_single_copy, src_region.blocks, dst_region.blocks))
        mark_box_dirty(dst_world, dimension, dst_region)
        return

    cx = ((dst_region.max_x + dst_region.min_x) >> 1)
    cy = ((dst_region.max_y + dst_region.min_y) >> 1)  # Paste is from-centre
    cz = ((dst_region.max_z + dst_region.min_z) >> 1)
    dst_region_midpoint = (cx, cy, cz)

    structure = ImmutableStructure.from_level(
        src_world, SelectionGroup(src_region), dimension)

    dst_world.paste(structure, structure.dimensions[0], SelectionGroup(src_region),
                                    dimension, dst_region_midpoint)
    #progress_iter(clone_op, "copy")
    # print("marking as dirty")
    # mark_box_dirty(dst_world, dimension, dst_region)
    # print("did a copy")
    # for cx, cz in src_region.chunk_locations():
    #     print("copy chunk", cx, cz, "from", src_world.level_path)

levels = dict()
def load_word(name):
    if name in levels:
        return levels.get(name)
    source_level = amulet.load_level(
            os.path.join(worlds_input_path, name))
    levels[name] = source_level
    return source_level
def do_conversion(regions: List[LayerConfig]):
    level_out = amulet.load_level(os.path.join(worlds_output_path, "world"))
    regions.reverse()
    vspace = -(converter_confg.height - converter_confg.overlap)
    def do(region, s, region_file_box):

        count = itertools.count()

        sub_chunk_size = 512  # 16
        #output_mask = SelectionBox((-560, 0, -32), (-560+sub_chunk_size, 255, -32+sub_chunk_size))
        vo = s * vspace
        realspace_output_mask = SelectionBox(
            (-converter_confg.spacing/2,
             converter_confg.min_y + vo,
             -converter_confg.spacing/2),
            (converter_confg.spacing/2,
             converter_confg.max_y + vo,
             converter_confg.spacing/2))

        # add filter for testing
        if region_file_box:
            realspace_output_mask = realspace_output_mask.intersection(region_file_box)
        # realspace_output_mask = SelectionBox.create_chunk_box(
        #     0, 0, 512).intersection(realspace_output_mask)

        src_selection = region.src_selection
        offset = region.offset.cords()
        dst_selection: SelectionBox = src_selection.create_moved_box(offset)
        dst_selection = dst_selection.intersection(realspace_output_mask)
        src_selection = dst_selection.create_moved_box(offset, subtract=True)
        dst_selection = dst_selection.create_moved_box(
            (converter_confg.spacing * s, -vo, 0))
        cbs = list(src_selection.chunk_boxes(sub_chunk_size))

        if src_selection.volume == 0:
            # not my table
            return
        print("layer",s, region, len(cbs), src_selection.volume, dst_selection)
        source_level = load_word(region.world)

        offset = dst_selection.min_array - src_selection.min_array

        def sub_do(s, i):
            _, src_selection = s
            dst_selection = src_selection.create_moved_box(offset)
            # _, dst_selection = d
            print("region", i, src_selection, dst_selection)
            copy_region(source_level, src_selection,
                        level_out, dst_selection)

            # level_out.unload_unchanged()
            # source_level.unload()
        list(map(sub_do, cbs, count))
        
        #progress_iter(level_out.save_iter(), "save")
        #level_out.unload()
        #source_level.unload()

    # with ProcessPoolExecutor() as executor:
    total_size = SelectionGroup()
    for region in regions:
        total_size = total_size.union(region.dst_selection)
    total_height = total_size.max_y - total_size.min_y
    num_slices = int(total_height / -vspace) + 1
    print(total_size.bounds, num_slices)
    region_file_boxes = {region_file_box for _,region_file_box in total_size.chunk_boxes()}
    for slice in range(num_slices):
        for region_file_box in region_file_boxes:
            for region in regions:
                do(region, slice, region_file_box)
            level_out.save()
            level_out.unload()
        #progress_iter(level_out.save_iter(), "save")
        
        # executor.
    # for region in regions:
    #     do(region)
    #     source_level=amulet.load_level(
    #         os.path.join(worlds_input_path, region.world))
    #     copy_region(source_level, region.src_selection,
    #                 level_out, region.dst_selection)
    #     source_level.close()
    progress_iter(level_out.save_iter(), "save")
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
