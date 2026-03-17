"""
CMPA数据保存脚本 - 年度整合版

将一年汛期（5-9月）数据保存到单个HDF文件中
保持原始0.05°分辨率，包含完整的时空维度信息

特点：
1. 单个HDF文件包含整个汛期数据
2. 包含时间、纬度、经度维度
3. 高效压缩（gzip level 9）
4. 分年份保存（2023.h5, 2024.h5）
"""

import numpy as np
import xarray as xr
import h5py
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings('ignore')

# 配置
CONFIG = {
    'data_root': '/data/data/CMPAS_P05_10MIN',
    'output_root': '/data/workspace/PyStepsDashu/data/cmpa_h5',

    # 数据参数
    'years': [2023, 2024],
    'months': [5, 6, 7, 8, 9],

    # 数据处理
    'invalid_value': 9999,
    'max_precip': 100.0,
    'fill_value': 0,

    # 压缩设置
    'compression': 'gzip',
    'compression_opts': 9,  # 最高压缩级别

    # 分块设置（优化大文件读写）
    'chunk_time': 10,   # 每次处理10个时间步
}


class CMPADataSaver:
    """CMPA数据年度整合保存器"""

    def __init__(self, config):
        self.config = config
        self.data_root = Path(config['data_root'])
        self.output_root = Path(config['output_root'])
        self.output_root.mkdir(parents=True, exist_ok=True)

    def process_year(self, year):
        """处理一年的数据并保存为单个HDF文件"""
        print("="*70)
        print(f"处理 {year} 年汛期数据")
        print("="*70)

        # 收集所有数据和时间信息
        all_data = []
        all_times = []
        file_count = 0

        for month in self.config['months']:
            month_str = f"{year}{month:02d}"
            month_dir = self.data_root / month_str

            if not month_dir.exists():
                print(f"  跳过 {month_str} (目录不存在)")
                continue

            # 获取该月的所有日期目录
            date_dirs = sorted([d for d in month_dir.iterdir() if d.is_dir()])
            print(f"\n{month_str}: {len(date_dirs)} 天")

            for date_dir in tqdm(date_dirs, desc=f"  读取", leave=False):
                # 解析日期
                try:
                    date = datetime.strptime(date_dir.name, "%Y%m%d")
                except ValueError:
                    continue

                # 只处理5-9月
                if date.month not in self.config['months']:
                    continue

                # 获取该日的所有GRB2文件并按时间排序
                grb2_files = sorted(list(date_dir.glob("*.GRB2")))

                if len(grb2_files) == 0:
                    continue

                # 读取每个文件
                for file_path in grb2_files:
                    try:
                        # 从文件名提取时间
                        time_str = self._extract_time_from_filename(file_path.name)
                        file_time = datetime.strptime(time_str, "%Y%m%d%H%M")

                        # 读取数据
                        ds = xr.open_dataset(str(file_path), engine='cfgrib')
                        data = ds['unknown'].values
                        ds.close()

                        # 处理无效值
                        data = np.where(data >= self.config['invalid_value'],
                                        self.config['fill_value'],
                                        data)

                        # 裁剪到合理范围
                        data = np.clip(data, 0, self.config['max_precip'])

                        all_data.append(data)
                        all_times.append(file_time)
                        file_count += 1

                    except Exception as e:
                        warnings.warn(f"处理文件失败 {file_path}: {e}")
                        continue

        if len(all_data) == 0:
            print(f"\n错误: {year} 年没有有效数据")
            return

        # 转换为numpy数组
        print(f"\n整理数据...")
        all_data = np.stack(all_data, axis=0)  # [time, lat, lon]

        # 获取空间维度信息
        n_times, n_lats, n_lons = all_data.shape
        lats = np.arange(n_lats) * 0.05 + 0.0
        lons = np.arange(n_lats) * 0.05 + 70.0

        print(f"数据形状: {all_data.shape}")
        print(f"时间范围: {len(all_times)} 个时间步")
        print(f"纬度范围: {n_lats} ({lats[0]:.2f}° - {lats[-1]:.2f}°)")
        print(f"经度范围: {n_lons} ({lons[0]:.2f}° - {lons[-1]:.2f}°)")

        # 保存为HDF文件
        output_file = self.output_root / f"{year}.h5"
        self._save_to_hdf(output_file, all_data, all_times, lats, lons)

        print(f"\n✓ 保存成功: {output_file}")
        self._print_file_info(output_file)

    def _extract_time_from_filename(self, filename):
        """从文件名提取时间字符串"""
        # 文件名格式: Z_SURF_C_BABJ_20230501002754_P_CMPA_FAST_CHN_0P05_10MIN-PRE-202305010000.GRB2
        parts = filename.split('_')

        # 找到时间部分 (通常在末尾附近)
        for i, part in enumerate(parts):
            if 'PRE-' in part:
                # 提取时间戳 (YYYYMMDDHHmm)
                time_str = part.split('-')[1].split('.')[0]
                return time_str

        # 如果没找到，使用备用方法
        if '2023' in filename or '2024' in filename:
            # 提取日期时间
            import re
            match = re.search(r'(\d{12})', filename)
            if match:
                return match.group(1)

        return filename[:12]  # 简化处理

    def _save_to_hdf(self, output_file, data, times, lats, lons):
        """保存数据到HDF文件"""
        print(f"\n保存到HDF文件...")

        with h5py.File(output_file, 'w') as f:
            # 创建数据集
            dset = f.create_dataset(
                'precipitation',
                data=data.astype('float32'),
                compression=self.config['compression'],
                compression_opts=self.config['compression_opts'],
                chunks=(self.config['chunk_time'], data.shape[1], data.shape[2])
            )

            # 添加维度信息
            f.create_dataset('time', data=np.array([t.timestamp() for t in times], dtype='float64'))
            f.create_dataset('latitude', data=lats.astype('float32'))
            f.create_dataset('longitude', data=lons.astype('float32'))

            # 添加属性
            dset.attrs['units'] = 'mm/h'
            dset.attrs['long_name'] = 'Precipitation Rate'
            dset.attrs['description'] = 'CMPA precipitation analysis'
            dset.attrs['resolution'] = '0.05 degree'
            dset.attrs['temporal_resolution'] = '10 minutes'

            # 时间属性
            f['time'].attrs['units'] = 'seconds since 1970-01-01'
            f['time'].attrs['long_name'] = 'Time'

            # 空间属性
            f['latitude'].attrs['units'] = 'degrees_north'
            f['latitude'].attrs['long_name'] = 'Latitude'

            f['longitude'].attrs['units'] = 'degrees_east'
            f['longitude'].attrs['long_name'] = 'Longitude'

            # 全局属性
            f.attrs['title'] = f'CMPA Precipitation Data - {times[0].year}'
            f.attrs['institution'] = 'CMA'
            f.attrs['source'] = 'China Meteorological Administration'
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['total_timesteps'] = len(times)
            f.attrs['date_range'] = f"{times[0].isoformat()} to {times[-1].isoformat()}"

    def _print_file_info(self, file_path):
        """打印文件信息"""
        size_mb = file_path.stat().st_size / (1024 * 1024)

        with h5py.File(file_path, 'r') as f:
            data = f['precipitation'][:]

            print(f"\n文件信息:")
            print(f"  路径: {file_path}")
            print(f"  大小: {size_mb:.2f} MB")
            print(f"  数据集形状: {data.shape}")
            print(f"  数据类型: {data.dtype}")
            print(f"  内存占用: {data.nbytes / (1024**3):.2f} GB")

            # 统计信息
            print(f"\n数据统计:")
            print(f"  最小值: {data.min():.2f} mm/h")
            print(f"  最大值: {data.max():.2f} mm/h")
            print(f"  平均值: {data.mean():.4f} mm/h")
            print(f"  有效降水像素: {(data > 0.1).sum()} ({(data > 0.1).sum() / data.size * 100:.4f}%)")

    def process_all_years(self):
        """处理所有年份"""
        print("="*70)
        print("CMPA年度数据整合保存")
        print("="*70)

        for year in self.config['years']:
            self.process_year(year)
            print()

        print("="*70)
        print("所有年份处理完成！")
        print("="*70)
        print(f"\n输出目录: {self.output_root}")


def main():
    """主函数"""
    saver = CMPADataSaver(CONFIG)
    saver.process_all_years()


if __name__ == "__main__":
    main()
