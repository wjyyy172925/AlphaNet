"""
clear_all_data.py
清除所有已下载和生成的数据，准备全新运行
"""

import shutil
from pathlib import Path
import pandas as pd

# 定义路径（与主代码保持一致）
RAW_DIR = Path('data/raw/baostock_daily')
MERGED_PATH = Path('df_merged.csv')
FEATURE_PATH = Path('df_merged_fe.csv')
FAILED_PATH = Path('failed_codes.csv')


def clear_all_data(confirm=True):
    """
    清除所有数据
    
    Parameters:
    -----------
    confirm : bool, default=True
        是否要求用户确认
    """
    
    # 统计要删除的文件
    files_to_delete = []
    total_size = 0
    
    # 1. 原始数据文件
    if RAW_DIR.exists():
        raw_files = list(RAW_DIR.glob('*.csv'))
        if raw_files:
            files_to_delete.extend(raw_files)
            # 计算总大小
            for f in raw_files:
                total_size += f.stat().st_size
        else:
            # 如果目录存在但没有csv文件，删除目录
            files_to_delete.append(RAW_DIR)
    
    # 2. 生成的文件
    generated_files = [MERGED_PATH, FEATURE_PATH, FAILED_PATH]
    for f in generated_files:
        if f.exists():
            files_to_delete.append(f)
            total_size += f.stat().st_size
    
    # 如果没有文件要删除
    if not files_to_delete:
        print("✅ 没有发现任何数据文件，无需清理")
        return True
    
    # 显示要删除的文件列表
    print("\n" + "="*60)
    print("📋 将要删除以下文件/目录：")
    print("="*60)
    
    for f in files_to_delete:
        if f.is_dir():
            size = sum(file.stat().st_size for file in f.glob('**/*') if file.is_file())
            print(f"  📁 {f} (包含 {len(list(f.glob('*')))} 个文件, {size / 1024 / 1024:.2f} MB)")
        else:
            size = f.stat().st_size
            print(f"  📄 {f} ({size / 1024:.2f} KB)")
    
    print("-"*60)
    print(f"总计: {len(files_to_delete)} 个文件/目录, {total_size / 1024 / 1024:.2f} MB")
    print("="*60)
    
    # 确认删除
    if confirm:
        response = input("\n⚠️  确认删除以上所有数据？(y/N): ")
        if response.lower() != 'y':
            print("❌ 操作已取消")
            return False
    
    # 执行删除
    try:
        for f in files_to_delete:
            if f.is_dir():
                shutil.rmtree(f)
                print(f"  ✅ 已删除目录: {f}")
            else:
                f.unlink()
                print(f"  ✅ 已删除文件: {f}")
        
        print("\n✅ 所有数据已清除完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 删除失败: {e}")
        return False


def clear_generated_only(confirm=True):
    """
    只清除生成的文件（保留原始数据）
    """
    files_to_delete = [MERGED_PATH, FEATURE_PATH, FAILED_PATH]
    existing_files = [f for f in files_to_delete if f.exists()]
    
    if not existing_files:
        print("✅ 没有生成文件需要清理")
        return True
    
    print("\n📋 将要删除以下生成的文件：")
    for f in existing_files:
        print(f"  📄 {f}")
    
    if confirm:
        response = input("\n⚠️  确认删除？(y/N): ")
        if response.lower() != 'y':
            print("❌ 操作已取消")
            return False
    
    for f in existing_files:
        f.unlink()
        print(f"  ✅ 已删除: {f}")
    
    print("✅ 生成文件清理完成！")
    return True


def clear_failed_only(confirm=True):
    """
    只清除失败记录和对应的原始文件
    """
    if not FAILED_PATH.exists():
        print("✅ 没有失败记录文件")
        return True
    
    # 读取失败列表
    failed_df = pd.read_csv(FAILED_PATH)
    failed_codes = failed_df['code'].tolist()
    
    print(f"\n📋 发现 {len(failed_codes)} 个失败记录")
    
    # 统计要删除的原始文件
    raw_files_to_delete = []
    for code in failed_codes:
        raw_file = RAW_DIR / f'{code}.csv'
        if raw_file.exists():
            raw_files_to_delete.append(raw_file)
    
    if raw_files_to_delete:
        print(f"将删除 {len(raw_files_to_delete)} 个原始数据文件")
        for f in raw_files_to_delete[:5]:  # 只显示前5个
            print(f"  📄 {f}")
        if len(raw_files_to_delete) > 5:
            print(f"  ... 还有 {len(raw_files_to_delete) - 5} 个文件")
    
    if confirm:
        response = input("\n⚠️  确认删除失败记录和对应的原始文件？(y/N): ")
        if response.lower() != 'y':
            print("❌ 操作已取消")
            return False
    
    # 删除原始文件
    for f in raw_files_to_delete:
        f.unlink()
        print(f"  ✅ 已删除: {f}")
    
    # 删除失败记录
    FAILED_PATH.unlink()
    print(f"  ✅ 已删除: {FAILED_PATH}")
    
    print("✅ 失败记录清理完成！")
    return True


def main():
    """交互式菜单"""
    print("\n" + "="*60)
    print("🧹 数据清理工具")
    print("="*60)
    print("\n请选择清理模式：")
    print("  1. 清除所有数据（原始数据 + 生成文件）")
    print("  2. 只清除生成的文件（保留原始数据）")
    print("  3. 只清除失败记录和对应的原始文件")
    print("  4. 查看数据统计")
    print("  0. 退出")
    
    choice = input("\n请输入选项 (0-4): ").strip()
    
    if choice == '1':
        clear_all_data()
    elif choice == '2':
        clear_generated_only()
    elif choice == '3':
        clear_failed_only()
    elif choice == '4':
        show_stats()
    elif choice == '0':
        print("退出")
    else:
        print("❌ 无效选项")


def show_stats():
    """显示数据统计"""
    print("\n" + "="*60)
    print("📊 数据统计")
    print("="*60)
    
    # 检查原始数据
    if RAW_DIR.exists():
        raw_files = list(RAW_DIR.glob('*.csv'))
        print(f"原始数据文件: {len(raw_files)} 个")
        if raw_files:
            total_size = sum(f.stat().st_size for f in raw_files)
            print(f"  总大小: {total_size / 1024 / 1024:.2f} MB")
    else:
        print("原始数据: 不存在")
    
    # 检查生成文件
    for path in [MERGED_PATH, FEATURE_PATH, FAILED_PATH]:
        if path.exists():
            size = path.stat().st_size
            print(f"{path.name}: {size / 1024:.2f} KB")
        else:
            print(f"{path.name}: 不存在")
    
    print("="*60)


if __name__ == '__main__':
    # 直接执行清理（带确认）
    # clear_all_data()
    
    # 或者使用交互式菜单
    main()