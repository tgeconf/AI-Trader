#!/usr/bin/env python3
"""
删除./data/agent_data/{}/log/{}下面的所有log.jsonl里每一条的"timestamp"字段
"""

import os
import json
import glob
from pathlib import Path

def remove_timestamp_from_log_files():
    """
    遍历所有agent_data目录下的log.jsonl文件，删除每一条记录中的timestamp字段
    """
    base_path = Path("./data/agent_data")
    
    if not base_path.exists():
        print(f"错误：目录 {base_path} 不存在")
        return
    
    # 查找所有log.jsonl文件
    log_files = list(base_path.glob("*/log/*/log.jsonl"))
    
    if not log_files:
        print("未找到任何log.jsonl文件")
        return
    
    print(f"找到 {len(log_files)} 个log.jsonl文件")
    
    processed_files = 0
    total_records = 0
    modified_records = 0
    
    for log_file in log_files:
        print(f"处理文件: {log_file}")
        
        try:
            # 读取文件内容
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                print(f"  文件为空，跳过")
                continue
            
            # 处理每一行
            modified_lines = []
            file_modified = False
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    modified_lines.append(line)
                    continue
                
                try:
                    # 解析JSON
                    data = json.loads(line)
                    total_records += 1
                    
                    # 检查是否有timestamp字段
                    if 'timestamp' in data:
                        # 删除timestamp字段
                        del data['timestamp']
                        modified_records += 1
                        file_modified = True
                        print(f"  第{line_num}行：删除了timestamp字段")
                    
                    # 重新序列化
                    modified_lines.append(json.dumps(data, ensure_ascii=False))
                    
                except json.JSONDecodeError as e:
                    print(f"  第{line_num}行：JSON解析错误 - {e}")
                    modified_lines.append(line)  # 保持原样
            
            # 如果有修改，写回文件
            if file_modified:
                with open(log_file, 'w', encoding='utf-8') as f:
                    for line in modified_lines:
                        f.write(line + '\n')
                print(f"  文件已更新")
                processed_files += 1
            else:
                print(f"  文件无需修改")
                
        except Exception as e:
            print(f"  处理文件时出错: {e}")
            continue
    
    print(f"\n处理完成！")
    print(f"处理了 {processed_files} 个文件")
    print(f"总共 {total_records} 条记录")
    print(f"修改了 {modified_records} 条记录")

if __name__ == "__main__":
    remove_timestamp_from_log_files()
