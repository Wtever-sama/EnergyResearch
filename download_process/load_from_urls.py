import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from tqdm import tqdm
import os
from datetime import datetime
from time import time as time_time

def set_logging(log_file_path='logs/download.log'):
    date = datetime.now().strftime('%Y%m%d')
    current_time = time_time()
    time_str = time.strftime('%H%M%S')
    dt = f'{date}_{time_str}'

    log_file_path = os.path.join('logs', f'download_{dt}.log')

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    # 创建日志记录器
    logger = logging.getLogger('download_logger')
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式器并添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger
    
logger = set_logging()

def load_from_url(url, save_dir, file_name, max_retries=20):
    for attempt in range(max_retries):
        try:
            # 完整文件路径
            full_path = os.path.join(save_dir, file_name)
            
            # 检查是否已经存在部分下载的文件
            initial_size = 0
            if os.path.exists(full_path):
                initial_size = os.path.getsize(full_path)
                logger.info(f'发现已存在的文件 {file_name}，大小: {initial_size} bytes')
                
                # 检查文件是否已经完全下载
                if is_file_fully_downloaded(url, save_dir, file_name):
                    logger.info(f'确认文件 {file_name} 已完全下载，跳过')
                    return full_path
            
            # 设置请求头支持断点续传
            headers = {}
            if initial_size > 0:
                headers['Range'] = f'bytes={initial_size}-'
            
            # 发送请求
            try:
                resp = requests.get(url, headers=headers, stream=True, timeout=60)
                # 特殊处理416错误
                if resp.status_code == 416:
                    logger.info(f'Range请求错误，文件 {file_name} 可能已完整')
                    return full_path
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 416:  # Range Not Satisfiable
                    logger.info(f'Range请求错误，文件 {file_name} 可能已完整')
                    return full_path
                else:
                    raise
            
            # 获取文件总大小
            total_size = initial_size
            if 'content-length' in resp.headers:
                total_size = initial_size + int(resp.headers['content-length'])
            elif initial_size == 0:
                # 尝试通过HEAD请求获取总大小
                try:
                    head_resp = requests.head(url, timeout=10)
                    if 'content-length' in head_resp.headers:
                        total_size = int(head_resp.headers['content-length'])
                except:
                    pass
            
            # 检查响应状态码
            if resp.status_code == 206:  # Partial Content
                logger.info(f'继续下载 {file_name} 从位置 {initial_size}')
                file_mode = 'ab'  # 追加模式
            elif resp.status_code == 200:  # OK
                if initial_size > 0:
                    logger.info(f'服务器不支持断点续传，重新下载 {file_name}')
                file_mode = 'wb'  # 覆盖模式
            else:
                resp.raise_for_status()
            
            logger.info(f'开始下载: {file_name} (总大小: {total_size} bytes)')
            
            # 确保保存目录存在
            os.makedirs(save_dir, exist_ok=True)
            
            # 创建进度条
            if total_size > 0:
                progress_bar = tqdm(
                    total=total_size,
                    initial=initial_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=file_name[:30]
                )
            else:
                progress_bar = tqdm(
                    initial=initial_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=file_name[:30]
                )
            
            # 下载文件并更新进度条
            try:
                with open(full_path, file_mode) as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:  # 过滤掉keep-alive chunks
                            f.write(chunk)
                            progress_bar.update(len(chunk))
                
                progress_bar.close()
                
                # 验证下载是否完整
                downloaded_size = os.path.getsize(full_path)
                if total_size > 0 and downloaded_size >= total_size:
                    logger.info('成功下载.NC文件：{} (100.0%)'.format(file_name))
                    return full_path
                elif total_size > 0:
                    percentage = (downloaded_size / total_size) * 100
                    logger.info('部分下载.NC文件：{} ({:.1f}%)'.format(file_name, percentage))
                    return full_path
                else:
                    logger.info('成功下载.NC文件：{} (大小: {} bytes)'.format(file_name, downloaded_size))
                    return full_path
                    
            except Exception as e:
                progress_bar.close()
                raise e
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:  # 不是最后一次尝试
                logger.warning(f'网络错误下载文件 {file_name} 时出错（第{attempt+1}次尝试）：{e} | 下载链接：{url}')
                time.sleep(2 ** attempt)  # 指数退避
                continue
            else:
                logger.error(f'网络错误下载文件 {file_name} 时出错（已尝试{max_retries}次）：{e} | 下载链接：{url}')
                return None
        except Exception as e:
            if attempt < max_retries - 1:  # 不是最后一次尝试
                logger.warning(f'下载文件 {file_name} 时出错（第{attempt+1}次尝试）：{e} | 下载链接：{url}')
                time.sleep(2 ** attempt)  # 指数退避
                continue
            else:
                logger.error(f'下载文件 {file_name} 时出错（已尝试{max_retries}次）：{e} | 下载链接：{url}')
                return None
    
    return None

def is_file_fully_downloaded(url, save_dir, file_name):
    """检查文件是否已经完全下载"""
    full_path = os.path.join(save_dir, file_name)
    
    try:
        head_resp = requests.head(url, timeout=10)
        if 'content-length' in head_resp.headers:
            total_size = int(head_resp.headers['content-length'])
            downloaded_size = os.path.getsize(full_path)
            if downloaded_size >= total_size:
                return True
    except:
        pass
    
    return False

def get_url_from_txt(file_path):
    '''
    Param: file_path | str
    Return: urls | list
    '''
    with open(file_path, 'r') as f:
        urls=[]
        for url in f.readlines():
            url = url.strip('\n')
            url = url.strip('* * ')
            urls.append(url)
    return urls

def get_txt_path(root_dir):
    '''
    Return:save_dir_load_txt| dict
        k: 保存目录，确保存在；
        v: txt文件路径'''

    save_dir_load_txt={}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if len(filenames) > 0:
            for fl in filenames:
                if fl.endswith('.txt') and fl.split('_')[0] in ['huss']:
                    save_dir_load_txt[f'{dirpath}']=(os.path.join(dirpath,fl))
    return save_dir_load_txt

def main():
    save_dir_load_txt=get_txt_path(root_dir = 'G:\\CanESM5-historical')

    logger.info('======下载.nc文件任务启动======')
    
    for key_sv_dir, val_txt_path in save_dir_load_txt.items():
        logger.info('   ——处理{}中的链接'.format(val_txt_path))
        urls = get_url_from_txt(val_txt_path)
        
        # 限制并发数，避免过多连接
        max_workers = min(len(urls), 10)  # 最多10个并发
        if max_workers < 1:
            max_workers = 1
        
        # 使用ThreadPoolExecutor和as_completed优化下载过程
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有下载任务
            future_to_url = {
                executor.submit(load_from_url, url, key_sv_dir, url.split('/')[-1]): url 
                for url in urls
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result is not None:
                        logger.info('       ——{} | 成功下载.NC文件：{}'.format(time.time(), result))
                    else:
                        logger.error('      ——{} | 下载失败.NC文件：{}'.format(time.time(), url))
                except Exception as e:
                    logger.error('      ——{} | 下载异常.NC文件：{} | 错误：{}'.format(time.time(), url, e))
                    
    logger.info('======下载.nc文件任务结束======')

main()