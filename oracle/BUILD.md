# Build de ygopro-core (libocgcore) sur macOS arm64

## Résumé

- Statut : **succès**. La bibliothèque partagée se compile et se charge via `ctypes`.
- Bibliothèque : `/Users/thomas/Documents/collab/ygo/oracle/ygopro-core/bin/release/libocgcore.dylib` (Mach-O arm64, release).
- Version affichée par `OCG_GetVersion` : **11.0**.

## Dépendances

- **Lua 5.4** : non installé via Homebrew. Il est vendu comme submodule git dans `lua/src` (commit `6e22fedb`, v5.4-beta), compilé avec le projet.
- **premake5** : installé avec `brew install premake` (version 5.0.0, bottle arm64).
- **Compilateur** : clang++ (Xcode command line tools), C++17.
- **Python 3** : `/usr/bin/python3` (module standard `ctypes`).

## Commandes utilisées

```sh
mkdir -p /Users/thomas/Documents/collab/ygo/oracle
cd /Users/thomas/Documents/collab/ygo/oracle
git clone --depth 1 --recursive https://github.com/edo9300/ygopro-core ygopro-core

brew install premake

cd ygopro-core
premake5 gmake2
make -C build ocgcoreshared config=release -j8

python3 /Users/thomas/Documents/collab/ygo/oracle/smoke.py
```

`make -C build ocgcoreshared config=release` produit `bin/release/libocgcore.dylib`. La cible `ocgcore` produirait une bibliothèque statique. L'option qui active la sortie des symboles est la définition `OCGCORE_EXPORT_FUNCTIONS`, appliquée par premake à la cible `ocgcoreshared`.

## Sortie du smoke test

```
OCG_GetVersion -> 11.0
```

Le script `smoke.py` charge la bibliothèque par chemin relatif à son propre emplacement, puis appelle `OCG_GetVersion(int* major, int* minor)`.

## Fonctions exportées (ocgapi.h)

Vérifiées avec `nm -gU` : les 13 symboles ci-dessous sont exportés.

```c
/*** CORE INFORMATION ***/
void OCG_GetVersion(int* major, int* minor);

/*** DUEL CREATION AND DESTRUCTION ***/
int OCG_CreateDuel(OCG_Duel* out_ocg_duel, const OCG_DuelOptions* options_ptr);
void OCG_DestroyDuel(OCG_Duel ocg_duel);
void OCG_DuelNewCard(OCG_Duel ocg_duel, const OCG_NewCardInfo* info_ptr);
void OCG_StartDuel(OCG_Duel ocg_duel);

/*** DUEL PROCESSING AND QUERYING ***/
int OCG_DuelProcess(OCG_Duel ocg_duel);
void* OCG_DuelGetMessage(OCG_Duel ocg_duel, uint32_t* length);
void OCG_DuelSetResponse(OCG_Duel ocg_duel, const void* buffer, uint32_t length);
int OCG_LoadScript(OCG_Duel ocg_duel, const char* buffer, uint32_t length, const char* name);

uint32_t OCG_DuelQueryCount(OCG_Duel ocg_duel, uint8_t team, uint32_t loc);
void* OCG_DuelQuery(OCG_Duel ocg_duel, uint32_t* length, const OCG_QueryInfo* info_ptr);
void* OCG_DuelQueryLocation(OCG_Duel ocg_duel, uint32_t* length, const OCG_QueryInfo* info_ptr);
void* OCG_DuelQueryField(OCG_Duel ocg_duel, uint32_t* length);
```

Les symboles dans le binaire (préfixe `_` propre à Mach-O) : `_OCG_CreateDuel`, `_OCG_DestroyDuel`, `_OCG_DuelGetMessage`, `_OCG_DuelNewCard`, `_OCG_DuelProcess`, `_OCG_DuelQuery`, `_OCG_DuelQueryCount`, `_OCG_DuelQueryField`, `_OCG_DuelQueryLocation`, `_OCG_DuelSetResponse`, `_OCG_GetVersion`, `_OCG_LoadScript`, `_OCG_StartDuel`.

## Problèmes rencontrés

- Aucun échec de build. Le premier essai a réussi sans correctif.
- Le README mentionne `./premake5` à la racine. Ici on utilise le premake5 de Homebrew sur le PATH, ce qui donne le même résultat.
- Les types `OCG_Duel`, `OCG_DuelOptions`, etc. sont définis dans `ocgapi_types.h`. Ce fichier n'a pas été lu en détail, car il n'est pas nécessaire pour ce smoke test.
