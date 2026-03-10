import os
from typing import List,Tuple
from collections.abc import Sequence
from numbers import Integral
import warnings
import math

def get_digit_length(n,max_decimal_places=None,*,point_1_bit=False,sign_1_bit=False):
    """
    获取数字长度
    Args:
        n: 整数或浮点数
        max_decimal_places: 最大小数位数，只对n为浮点数时有效，当其不为None且n为浮点数时函数会截断n到指定小数位
        point_1_bit: 小数点是否占一位,如为True占的这位会算在总长度上

    Returns:(整数部分长度, 小数部分长度, 总长度)


    """
    if isinstance(n,int):
        n = abs(n)
        if n == 0:
            length = 1
        else:
            length = math.floor(math.log10(n)) + 1
        if sign_1_bit and n < 0:
            return length,0,length+1
        return length,0,length
    elif isinstance(n,float):
        return get_float_parts_length(n,max_decimal_places,point_1_bit=point_1_bit)

def get_float_parts_length(num, max_decimal_places=None,*,point_1_bit=False):
    """
    获取浮点数的整数部分长度、小数部分长度和总长度
    Args:
        num: 浮点数
        max_decimal_places: 可选，指定保留的小数位数（使用 round 格式化）
        point_1_bit: 小数点是否占一位,如为True占的这位会算在总长度上

    Returns: (整数部分长度, 小数部分长度, 总长度)
    """
    assert isinstance(num,float)
    # 格式化：可选指定小数位数
    if max_decimal_places is not None:
        s = f"{num:.{max_decimal_places}f}"
    else:
        s = str(num)

    # 处理科学计数法（如 1.23e-05）
    if 'e' in s or 'E' in s:
        s = f"{num:.10f}".rstrip('0').rstrip('.')  # 转为普通小数表示

    # 处理负号
    has_minus = s.startswith('-')
    if has_minus:
        s = s[1:]

    # 拆分整数和小数部分
    if '.' in s:
        int_part, frac_part = s.split('.')
        frac_part = frac_part.rstrip('0')  # 去除末尾无效0
        if frac_part == '':
            frac_part = '0'  # 如 5.000 → 小数部分视为 '0'
    else:
        int_part, frac_part = s, '0'

    int_len = len(int_part)
    frac_len = len(frac_part)
    total_len = int_len + frac_len + 1  # +1 是小数点
    if has_minus:
        total_len += 1  # 负号
    if point_1_bit:
        total_len += 1

    return int_len, frac_len, total_len

def auto_unit(num:int|float,
              units:Tuple[str]=('','K','M','B','T','Qa','Qi'),entry_rate:int|List[int]=1000,*,
              base_unit_index:int=0,return_dict:bool=False) -> str|dict:
    """
    自动单位转换
    Args:
        num: 需要转换单位的原始数值
        units: 有效单位表
        entry_rate: 单位进率，应为整数或序列。为整数时则恒定进率；为序列时会取对应索引的值为进率。注意，其长度应比有效单位表小1。
        base_unit_index: 原始数值的单位在有效单位表中的索引
        return_dict: 是否返回字典

    Returns:带单位的字符串或是字典{"Value":current_num,"Unit":current_unit}

    """

    # 合法性检测
    if not isinstance(num,int) and not isinstance(num,float):
        try:
            num = float(num)
        except Exception as e:
            raise TypeError(f"num类型错误:num应为整型、浮点型或数字字符串，而非{type(num)}(num为{num})")
    if isinstance(entry_rate,int):
        l = len(units) - 1
        entry_rate = [entry_rate for _ in range(l)]
    elif isinstance(entry_rate, Sequence) and not isinstance(entry_rate, (str, bytes, bytearray)):
        if not all(isinstance(x, Integral) and not isinstance(x,bool) for x in entry_rate):
            raise ValueError('entry_rate 为序列时其元素必须为整型数.')
        if len(entry_rate) < len(units) - 1:
            raise ValueError(f'进率entry_rate的长度小于单位units的长度减一，进率entry_rate的长度应等于单位units的长度减一.')
        elif len(entry_rate) > len(units) - 1:
            warnings.warn(f"进率entry_rate的长度过长,单位units的长度为{len(units)}但进率entry_rate的长度为{len(entry_rate)}。", RuntimeWarning)
            entry_rate = entry_rate[:len(units) - 1]
    assert 0 <= base_unit_index < len(units) and isinstance(base_unit_index, int)
    current_unit = units[base_unit_index] # 当前单位
    current_unit_idx = base_unit_index # 当前单位在units中的索引
    original_sign = -1 if num < 0 else 1 # 数值符号
    current_num = abs(num) # 只使用绝对值进行转换
    # 只对current_unit_idx在[1, len(entry_rate)-1)的情况进行进退位
    while 0<=current_unit_idx<len(units)-1:
        # 根据current_num与entry_rate[current_unit_idx]或1的关系判断是进位还是退位
        er = entry_rate[current_unit_idx]
        if current_num >= er:
            current_unit_idx += 1
            current_num /= er
            current_unit = units[current_unit_idx]
        elif current_num < 1:
            if current_unit_idx == 0: # 如果current_unit_idx为0且需要退位则直接跳出
                break
            current_unit_idx -= 1
            current_num *= entry_rate[current_unit_idx]
            current_unit = units[current_unit_idx]
        else:
            break
    # 应用原始符号
    current_num *= original_sign
    if return_dict:
        return {"Value":current_num,"Unit":current_unit}
    return f"{current_num}{current_unit}"

def auto_number_length(num:int|float, int_places:int=4, decimal_places:int=2, *,
                       max_length_num: None | int=None, point_1_bit=False
                       ):
    """
    自动数字长度函数，可以通过int_places和decimal_places手动控制数字长度，也可以通过输入最长数字自动同步长度
    Args:
        num: 数字或数字字符串
        int_places: 整数位数
        decimal_places: 小数位数
        max_length_num: 最长数字
        point_1_bit: 小数点是否占一位

    Returns:指定长度的数字字符串

    """
    assert int_places >= 0
    assert decimal_places >= 0
    if not isinstance(num,int) and not isinstance(num,float):
        try:
            num = int(num)
        except:
            try:
                num = float(num)
            except:
                raise TypeError("自动数字长度函数auto_number_length输入的数字必须为整型或浮点型数字(字符串)")
    if max_length_num is None:
        total_width = int_places + decimal_places
        if isinstance(num, int):
            fmt = f"0{int_places}d" if int_places else "d"
            return format(num, fmt)
        else:  # float
            if point_1_bit:
                fmt = f"0{total_width}.{decimal_places}f" if int_places else f".{decimal_places}f"
            else:
                fmt = f"0{total_width + 1}.{decimal_places}f" if int_places else f".{decimal_places}f"
            return format(num, fmt)
    else:
        if not isinstance(max_length_num, int):
            try:
                max_length_num = int(max_length_num)
            except Exception as e:
                try:
                    max_length_num = float(max_length_num)
                except Exception as e:
                    raise TypeError(f"自动数字长度函数auto_number_length的max_num只能是数字、数字字符串或None，不能是{type(num)}")
        int_places,decimal_places,_ = get_digit_length(max_length_num)
        return auto_number_length(num,int_places=int_places,decimal_places=decimal_places,point_1_bit=point_1_bit)



if __name__ == "__main__":
 pass


