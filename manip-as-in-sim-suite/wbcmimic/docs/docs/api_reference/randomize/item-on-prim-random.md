

```python 
EventTerm(
        func=ResetItemOnPrim().__call__,
        mode="reset",
        params={
            "random_item_cfgs": [SceneEntityCfg("cup"), SceneEntityCfg("wine_bottle")],
            "place_prim_cfg": SceneEntityCfg("table"),
            'rotation_range': {'roll': (0, 0), 'pitch': (0, 0), 'yaw': (0, 180)},
            "height_offset": 0.1,
            "use_prim_max_height": True,
            "max_attemp_times": 20,
        },
    )

```
- `random_item_cfgs`: The list of item configurations to be randomized
- `place_prim_cfg`: The configuration of the place item's prim (e.g. the table)
- `rotation_range`: The range of the rotation
- `height_offset`: The offset of the height
- `use_prim_max_height`: Whether to use the max height of the primitive
- `max_attemp_times`: The maximum number of attempts




