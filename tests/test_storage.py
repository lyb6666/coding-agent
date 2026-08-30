"""storage.py 的单元测试：验证会话历史的保存 / 加载 / 清除。"""
import coding_agent.storage as storage


def test_history_save_load_clear(tmp_path):
    working_dir = tmp_path / "project"
    working_dir.mkdir()
    messages = [
        {"role": "system", "content": "你是编程智能体"},
        {"role": "user", "content": "写一个脚本"},
    ]

    assert storage.load_history(working_dir) is None  # 初始无历史

    storage.save_history(working_dir, messages)
    assert (working_dir / storage.HISTORY_FILENAME).exists()  # 历史文件就在文件夹内
    assert storage.load_history(working_dir) == messages  # 保存后可加载

    storage.clear_history(working_dir)
    assert storage.load_history(working_dir) is None  # 清除后无历史


def test_history_is_per_working_directory(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()

    storage.save_history(a, [{"role": "user", "content": "a 的对话"}])
    storage.save_history(b, [{"role": "user", "content": "b 的对话"}])

    # 不同文件夹的历史互不干扰，各自有自己的历史文件
    assert storage.load_history(a)[0]["content"] == "a 的对话"
    assert storage.load_history(b)[0]["content"] == "b 的对话"
    assert (a / storage.HISTORY_FILENAME).exists()
    assert (b / storage.HISTORY_FILENAME).exists()
