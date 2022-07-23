from __future__ import print_function
from pymol import cmd

def command(pdbId=None, pdbFile=None, position=None, chain=None, bg_color=None, helix_color=None,
            bsheet_color=None, loop_color=None, other_color=None, sphe_size=None, **kwargs):
    """
    residual position setting
    """
    if position==None:
        print("Check your input!")
    if chain==None:
        chain="A"

    """
    Color setting
    """
    if bg_color==None:
        bg_color="white"
    if helix_color==None:
        helix_color="yellow"
    if bsheet_color==None:
        bsheet_color="green"
    if loop_color==None:
        loop_color="orange"
    if other_color==None:
        other_color="grey"
    if sphe_size==None:
        sphe_size=3

    if pdbId:
        cmd.fetch(pdbId)
    elif pdbId==None and pdbFile:
        cmd.load(pdbFile)
    else:
        print("Check your PDB input, either give a PDB id or PDB file with full directory!")

    cmd.bg_color(bg_color)
    cmd.color(helix_color, 'ss h')
    cmd.color(bsheet_color, 'ss s')
    cmd.color(loop_color, 'ss l')
    cmd.color(other_color, 'ss ""')

    cmd.select('myresi', 'chain '+ chain + ' and resi %s' % position)

    #def plotResi(selection):
    model = cmd.get_model('myresi')
    x, y, z = 0, 0, 0
    for a in model.atom:
        x += a.coord[0]
        y += a.coord[1]
        z += a.coord[2]
    pos=[x/len(model.atom), y/len(model.atom), z/len(model.atom)]
    print(pos)

    object = cmd.get_legal_name("myrei")
    object = cmd.get_unused_name(object + "_COM", 0)
    cmd.delete(object)
    cmd.pseudoatom(object, pos=pos)
    cmd.show("spheres", object)
    cmd.set ("sphere_scale", sphe_size, selection=object)
# plotResi('myresi')
#