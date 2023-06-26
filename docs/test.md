## 参数

1. `-h   --help`
    + 输出帮助信息并结束程序（此参数权重最大）
2. `-d   --device`
    + 运行设备如： `cuda:0`，`cuda:1`，`cuda:2` 和 `cpu`等
    + 默认 `cpu`
3. `-i   --input`
    + 输入文件路径或摄像头序号0和1等
    + 默认 `0`
4. `-o   --output`
    + 输出文件名（不需要后缀）
    + 默认为空
5. `-v   --version`
    + 使用Yolo的版本`[v5, v6, v7, v8]`中选一
    + 默认 `v8`
6. `-mh  --model_height`
    + 模型处理图片高度
    + 默认 640
7. `-mw  --model_width`
    + 模型处理图片宽度
    + 默认 640
8. `-wh  --window_height`
    + 显示窗口高度（输入为摄像头时也表示捕捉图片高度）
    + 默认 640
9. `-ww  --window_width`
    + 显示窗口宽度（输入为摄像头时也表示捕捉图片宽度）
    + 默认 640
10. `-mp  --model_path`
    + 模型路径
    + 默认 `yolov8n.cpu.torchscript`
11. `-lp  --label_path`
    + 标签路径
    + 默认 `coco.txt`
12. `-r   --roi`
    + 目标区域检测 `[on, in]` 二选一，不携带参数默认为`in`
    + 默认 `false`
13. `-ic  --is_close`
    + 是否关闭窗口
    + 默认 `false`
14. `-ih  --is_half`
    + 是否半精度
    + 默认 `false`
15. `-ii  --is_image`
    + 是否输入为图片
    + 默认 `false`
