"""Une recette Lua de fusion conserve une inclusion et une exclusion de type."""
from pathlib import Path
from unittest.mock import patch
import sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'translate'),str(ROOT/'engine')]
import translate as T,core,cards
source='''local s,id=GetID()
function s.initial_effect(c)
 c:EnableReviveLimit()
 Fusion.AddProcMix(c,false,true,s.mat,s.mat)
end
function s.mat(c)
 return c:IsType(TYPE_MONSTER) and not c:IsType(TYPE_EFFECT)
end
'''
with tempfile.TemporaryDirectory() as folder:
    Path(folder,'c990000004.lua').write_text(source)
    with patch.object(T,'OFFICIAL',folder):effects,notes,recipe,descs=T.translate(990000004)
assert recipe[0]=='fusion' and len(recipe[1])==2
for material in recipe[1]:
    assert material[core.F_TYPE]==core.TYPE_MONSTER and material[core.F_NOTTYPE]==cards.TYPE_EFFECT
print('Fusion.AddProcMix : les deux filtres combinent inclusion et exclusion')
