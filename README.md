# A_command templates generated from monitor.zip PPM frames

Directory structure is ready for `template_icon_digit` backend:

```
templates/digits/{0,1,2,3,4,5}/*.png
templates/icons/{flange_nut,gear_ring,spacer_ring,hex_nut,dome_nut}.png
```

Digits are normalized 48x48 black-on-white PNGs extracted from the supplied PPM frames.
Icon templates are raw camera crops resized to 96x72.

Important: digit folders `4` and `5` are intentionally empty because the provided frames did not contain clear examples of 4 or 5. When those counts appear, save `/tmp/monitor_ocr_debug/row*_digit_norm.png` and add them to the corresponding folder.

Use:

```bash
mkdir -p /ws/src/monitor_ocr_a/monitor_ocr_a/templates
cp -r templates/digits /ws/src/monitor_ocr_a/monitor_ocr_a/templates/
cp -r templates/icons /ws/src/monitor_ocr_a/monitor_ocr_a/templates/
```

Then run with explicit params if supported:

```bash
-p digit_template_dir:=/ws/src/monitor_ocr_a/monitor_ocr_a/templates/digits \
-p icon_template_dir:=/ws/src/monitor_ocr_a/monitor_ocr_a/templates/icons
```


## Update: synthetic 4/5 templates

This package includes synthetic fallback templates for digit folders `4` and `5`. I did not find clear real camera examples of 4 or 5 in the supplied PPM frames, so these are generated in the same normalized 48x48 black-on-white format. They are intended only as a starter fallback. When A_command shows quantity 4 or 5, save the real `/tmp/monitor_ocr_debug/row*_digit_norm.png` crops and add them to `templates/digits/4/` or `templates/digits/5/` for better accuracy.
