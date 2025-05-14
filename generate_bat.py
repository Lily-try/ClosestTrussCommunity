def generate_bat_script(input_file='.\Dataset\cora\query.txt', output_file='run.bat'):
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='\r\n') as f_out:
        # 读取查询总数
        n = int(f_in.readline().strip())
        
        # 生成每个查询命令
        for _ in range(n):
            line = f_in.readline().strip()
            if not line:
                break
                
            # 分割参数
            parts = line.split()
            t = parts[0]
            nodes = ' '.join(parts[1:])
            
            # 写入批处理命令（使用Windows路径格式）
            f_out.write(f'.\\main.exe cora {t} {nodes}\n')

if __name__ == "__main__":
    generate_bat_script()
    print("批处理文件已生成：run.bat")