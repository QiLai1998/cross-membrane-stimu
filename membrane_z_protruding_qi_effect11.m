% 创建网格数据
[x,y] = meshgrid(-8:0.3:8);  % 网格数据
frames = 50;  % 总帧数

% ===== 参数配置 =====
show_grid = false;       % 是否显示网格线
bottom_alpha = 0.35;     % 底层膜透明度
top_alpha = 0.35;        % 顶膜透明度
bottom_base_height = -1.45; % 底膜基础高度
top_base_height = 1.45;  % 顶膜基础高度
middle_base_height = -16.0; % 中间膜基础高度
fusion_threshold = 0.65; % 融合阈值
arrow_scale = 1.0;      % 箭头缩放因子
speed_threshold = 0.1; % 最小速度阈值（过滤小箭头）
max_arrow_speed = 1.5;  % 最大速度阈值（过滤异常大箭头）
max_speed = max_arrow_speed; % 颜色映射最大值（与最大箭头速度一致）
arrow_density = 2;      % 箭头密度
arrow_line_width = 0.6; % 箭头线条粗细
arrow_head_size = 0.6;  % 箭头头部大小比例
center_region_threshold = 0.3;  % 中心区域阈值
middle_amplitude = 0.1; % 中间膜中心突出幅度
middle_height = 30;    % 中间膜中心突出高度
middle_alpha = 0.6;    % 中间膜透明度
middle_noise_strength = 0.1; % 中间膜噪声强度

% 新增：椭球体参数配置
ellipsoid_a = 12;       % 椭球体X轴半轴长度
ellipsoid_b = 12;       % 椭球体Y轴半轴长度
ellipsoid_c = 12;       % 椭球体Z轴半轴长度（控制扁平程度）
ellipsoid_offset = 0.5;  % 椭球体整体偏移量（控制基础高度）

% 新增：三个膜的箭头显示控制参数
show_arrow_bottom = false;   % 控制底膜箭头显示
show_arrow_top = false;      % 控制顶膜箭头显示
show_arrow_middle = true;   % 控制中间膜箭头显示

% 新增：初始帧和结束帧突起调整参数
initial_frame_amplitude_scale = 0.5;  % 初始帧突起幅度缩放
initial_frame_shift_scale = 0.5;      % 初始帧垂直偏移缩放
final_frame_amplitude_scale = 0.5;    % 结束帧突起幅度缩放
final_frame_shift_scale = 0.5;        % 结束帧垂直偏移缩放
% ====================

% 预计算随机种子序列
rng(42);  % 固定随机种子
noise_seeds = rand(frames, 5);  % 每帧5个随机种子

% 存储各帧的高度数据
z_bottom_all = zeros(frames, size(x,1), size(x,2));
z_top_all = zeros(frames, size(x,1), size(x,2));
z_middle_all = zeros(frames, size(x,1), size(x,2)); % 中间膜高度数据
membrane_distances = zeros(frames, 1); % 每帧膜间距

% 预计算所有帧的高度数据
fprintf('预计算高度数据...\n');
for j = 1:frames
    % 时间相位
    phase_bottom = 2*pi * j/frames;
    phase_middle = 2*pi * j/frames + pi/3; % 中间膜相位偏移
    
    % 底层膜随机律动（复用噪声）
    t = j/frames;
    noise = 0;
    for k = 1:5  % 多频率噪声叠加，产生不规则律动
        freq = 2^k;
        amp = 0.2/k;
        noise = noise + amp * sin(freq*x + noise_seeds(j,k)*100) .* sin(freq*y + noise_seeds(j,k)*200);
    end
    membrane_shift = 0.3 * noise;  % 底膜的基础律动
    
    % 中心区域与边缘区域区分
    distance_from_center = sqrt(x.^2 + y.^2);
    pulse_influence = exp(-(distance_from_center/2).^2); % 中心区域影响（中心大，边缘小）
    membrane_influence = 1 - pulse_influence; % 边缘区域影响（边缘大，中心小）
    
    % 中心基础形状（确保为正，参考底膜）
    base = sin(sqrt(x.^2+y.^2))./sqrt(x.^2+y.^2+eps);
    base_positive = abs(base);  % 确保基础形状为正，避免负值导致向下突出
    
    % 底膜Z值计算
    bottom_amplitude = 2 * (1 + 0.5*sin(phase_bottom));
    bottom_vertical_shift = 0.5 * sin(phase_bottom) + 0.5;
    
    z_bottom = bottom_base_height + ...
        membrane_shift .* membrane_influence + ...  % 边缘不规则律动
        (base_positive * bottom_amplitude + bottom_vertical_shift) .* pulse_influence; % 中心向上突出
    
    % 顶膜效果
    top_membrane_base = top_base_height;
    top_membrane_dip = -1.0 * base_positive * bottom_amplitude;
    top_noise = -membrane_shift;
    z_top = top_membrane_base + top_membrane_dip + top_noise;
    
    % 中间膜效果
    ellipsoid_term = 1 - (x./ellipsoid_a).^2 - (y./ellipsoid_b).^2;
    ellipsoid_term(ellipsoid_term < 0) = 0;  % 确保根号内非负
    ellipsoid_base = sqrt(ellipsoid_term) * ellipsoid_c;  % 椭球体表面高度
    
    middle_min_offset = 0.5;  % 最小向上偏移量
    middle_oscillation = 0.3 * sin(phase_middle);  % 周期性振荡
    middle_base_offset = middle_min_offset + abs(middle_oscillation);  % 总基础偏移
    
    middle_center_peak = middle_amplitude * base_positive * middle_height * (1 + 0.5*sin(phase_middle));
    middle_edge_shift = middle_noise_strength * noise;  % 边缘区域噪声
    
    z_middle = middle_base_height + ...
        ellipsoid_base + ...  % 椭球体基础形状
        middle_base_offset + ...  % 基础向上偏移
        middle_center_peak .* pulse_influence + ...  % 中心突出效果
        middle_edge_shift .* membrane_influence;  % 边缘动态效果
    
    % 融合效果
    distance_between_membranes = z_top - z_bottom;
    membrane_distances(j) = mean(distance_between_membranes(:));
    fusion_mask = distance_between_membranes < fusion_threshold;
    
    if any(fusion_mask(:))
        z_fusion = (z_top + z_bottom) / 2;
        z_bottom(fusion_mask) = z_fusion(fusion_mask);
        z_top(fusion_mask) = z_fusion(fusion_mask);
    end
    
    % 存储高度数据
    z_bottom_all(j,:,:) = z_bottom;
    z_top_all(j,:,:) = z_top;
    z_middle_all(j,:,:) = z_middle;
    
    fprintf('预计算帧 %d/%d\r', j, frames);
end
fprintf('\n');

% 找到膜间距最大的起始帧
[~, start_frame_idx] = max(membrane_distances);
fprintf('起始帧: %d (膜间距最大)\n', start_frame_idx);

% 计算结束帧索引（循环播放时需要与起始帧衔接）
end_frame_idx = mod(start_frame_idx + frames - 1 - 1, frames) + 1;
fprintf('结束帧: %d\n', end_frame_idx);

% 调整初始帧的中央突起高度
frame_idx = start_frame_idx;
phase_bottom = 2*pi * frame_idx/frames;

% 初始帧使用降低后的突起参数
reduced_amplitude = 2 * (1 + 0.5*sin(phase_bottom)) * initial_frame_amplitude_scale;
reduced_shift = (0.5 * sin(phase_bottom) + 0.5) * initial_frame_shift_scale;

% 重新计算中心形状
distance_from_center = sqrt(x.^2 + y.^2);
pulse_influence = exp(-(distance_from_center/2).^2);
base = sin(sqrt(x.^2+y.^2))./sqrt(x.^2+y.^2+eps);
base_positive = abs(base);

% 重新计算底膜高度（降低突起）
z_bottom_initial = bottom_base_height + ...
    (squeeze(z_bottom_all(frame_idx,:,:)) - bottom_base_height - ...
    (base_positive * 2 * (1 + 0.5*sin(phase_bottom)) + (0.5 * sin(phase_bottom) + 0.5)) .* pulse_influence) + ...
    (base_positive * reduced_amplitude + reduced_shift) .* pulse_influence;

% 重新计算顶膜高度（对应降低凹陷）
z_top_initial = squeeze(z_top_all(frame_idx,:,:)) + ...
    1.0 * base_positive * (2 * (1 + 0.5*sin(phase_bottom)) - reduced_amplitude);

% 更新存储的高度数据
z_bottom_all(frame_idx,:,:) = z_bottom_initial;
z_top_all(frame_idx,:,:) = z_top_initial;

% 调整结束帧的中央突起高度（关键新增部分）
frame_idx = end_frame_idx;
phase_bottom = 2*pi * frame_idx/frames;

% 结束帧使用降低后的突起参数
reduced_amplitude = 2 * (1 + 0.5*sin(phase_bottom)) * final_frame_amplitude_scale;
reduced_shift = (0.5 * sin(phase_bottom) + 0.5) * final_frame_shift_scale;

% 重新计算底膜高度（降低突起）
z_bottom_final = bottom_base_height + ...
    (squeeze(z_bottom_all(frame_idx,:,:)) - bottom_base_height - ...
    (base_positive * 2 * (1 + 0.5*sin(phase_bottom)) + (0.5 * sin(phase_bottom) + 0.5)) .* pulse_influence) + ...
    (base_positive * reduced_amplitude + reduced_shift) .* pulse_influence;

% 重新计算顶膜高度（对应降低凹陷）
z_top_final = squeeze(z_top_all(frame_idx,:,:)) + ...
    1.0 * base_positive * (2 * (1 + 0.5*sin(phase_bottom)) - reduced_amplitude);

% 更新存储的高度数据
z_bottom_all(frame_idx,:,:) = z_bottom_final;
z_top_all(frame_idx,:,:) = z_top_final;

% ===== 交互式调整部分（核心修改）=====
fprintf('正在准备交互式调整视图...\n');

% 明确设置窗口大小
fig_width = 1000;
fig_height = 800;

% 获取第一帧数据用于预览
frame_idx = start_frame_idx;
next_frame_idx = mod(start_frame_idx, frames) + 1;
dt = 1 / 15;  % 使用默认帧率计算

% 计算第一帧速度
z_bottom = squeeze(z_bottom_all(frame_idx,:,:));
z_top = squeeze(z_top_all(frame_idx,:,:));
z_middle = squeeze(z_middle_all(frame_idx,:,:));

vz_bottom_raw = squeeze(z_bottom_all(next_frame_idx,:,:) - z_bottom_all(frame_idx,:,:)) / dt;
vz_top_raw = squeeze(z_top_all(next_frame_idx,:,:) - z_top_all(frame_idx,:,:)) / dt;
vz_middle_raw = squeeze(z_middle_all(next_frame_idx,:,:) - z_middle_all(frame_idx,:,:)) / dt;

vz_bottom = max(min(vz_bottom_raw, max_arrow_speed), -max_arrow_speed);
vz_top = max(min(vz_top_raw, max_arrow_speed), -max_arrow_speed);
vz_middle = max(min(vz_middle_raw, max_arrow_speed), -max_arrow_speed);

% 计算法向量（用于箭头方向）
distance_from_center = sqrt(x.^2 + y.^2);
pulse_influence = exp(-(distance_from_center/2).^2);

[dx_bottom, dy_bottom] = gradient(z_bottom, 0.3, 0.3);
normal_mag_bottom = sqrt(1 + dx_bottom.^2 + dy_bottom.^2);
nx_bottom = -dx_bottom ./ normal_mag_bottom;
ny_bottom = -dy_bottom ./ normal_mag_bottom;
nz_bottom = 1 ./ normal_mag_bottom;

[dx_top, dy_top] = gradient(z_top, 0.3, 0.3);
normal_mag_top = sqrt(1 + dx_top.^2 + dy_top.^2);
nx_top = -dx_top ./ normal_mag_top;
ny_top = -dy_top ./ normal_mag_top;
nz_top = 1 ./ normal_mag_top;

[dx_middle, dy_middle] = gradient(z_middle, 0.3, 0.3);
normal_mag_middle = sqrt(1 + dx_middle.^2 + dy_middle.^2);
nx_middle = -dx_middle ./ normal_mag_middle;
ny_middle = -dy_middle ./ normal_mag_middle;
nz_middle = 1 ./ normal_mag_middle;

% 下采样箭头（控制箭头密度）
sparse_idx = 1:arrow_density:size(x,1);
M = length(sparse_idx);

x_sparse = x(sparse_idx, sparse_idx);
y_sparse = y(sparse_idx, sparse_idx);
z_bottom_sparse = z_bottom(sparse_idx, sparse_idx);
z_top_sparse = z_top(sparse_idx, sparse_idx);
z_middle_sparse = z_middle(sparse_idx, sparse_idx);

% 法向量和速度下采样
nx_bottom_sub = nx_bottom(sparse_idx, sparse_idx);
ny_bottom_sub = ny_bottom(sparse_idx, sparse_idx);
nz_bottom_sub = nz_bottom(sparse_idx, sparse_idx);
vz_bottom_sub = vz_bottom(sparse_idx, sparse_idx);

nx_top_sub = nx_top(sparse_idx, sparse_idx);
ny_top_sub = ny_top(sparse_idx, sparse_idx);
nz_top_sub = nz_top(sparse_idx, sparse_idx);
vz_top_sub = vz_top(sparse_idx, sparse_idx);

nx_middle_sub = nx_middle(sparse_idx, sparse_idx);
ny_middle_sub = ny_middle(sparse_idx, sparse_idx);
nz_middle_sub = nz_middle(sparse_idx, sparse_idx);
vz_middle_sub = vz_middle(sparse_idx, sparse_idx);

% 中心区域掩码（过滤边缘区域）
pulse_influence_sub = pulse_influence(sparse_idx, sparse_idx);
center_mask_bottom = pulse_influence_sub > center_region_threshold;
center_mask_top = pulse_influence_sub > center_region_threshold;
center_mask_middle = pulse_influence_sub > center_region_threshold;

% 速度掩码（过滤过小/过大的箭头）
speed_bottom = sqrt(vz_bottom_sub.^2);
speed_top = sqrt(vz_top_sub.^2);
speed_middle = sqrt(vz_middle_sub.^2);
speed_mask_bottom = (speed_bottom > speed_threshold) & (speed_bottom <= max_arrow_speed);
speed_mask_top = (speed_top > speed_threshold) & (speed_top <= max_arrow_speed);
speed_mask_middle = (speed_middle > speed_threshold) & (speed_middle <= max_arrow_speed);

% 复合掩码（同时满足中心区域和速度条件）
bottom_mask = center_mask_bottom & speed_mask_bottom;
top_mask = center_mask_top & speed_mask_top;
middle_mask = center_mask_middle & speed_mask_middle; 


% 交互式调整视图（后续代码与之前完全一致，省略）
fig = figure('Position', [100 100 fig_width fig_height]);
title('请调整视图（旋转、缩放），完成后按回车键继续');
hold on;

% 绘制三层膜和箭头
h_middle = surf(x, y, z_middle);
set(h_middle, 'FaceAlpha', middle_alpha, 'FaceColor', '#E0FFFF');
if show_grid
    set(h_middle, 'EdgeAlpha', middle_alpha*0.5, 'EdgeColor', [0 0 0]);
else
    set(h_middle, 'EdgeAlpha', 0);
end

h_bottom = surf(x, y, z_bottom);
set(h_bottom, 'FaceAlpha', bottom_alpha, 'FaceColor', '#C9A0DC');
if show_grid
    set(h_bottom, 'EdgeAlpha', bottom_alpha*0.5, 'EdgeColor', [0 0 0]);
else
    set(h_bottom, 'EdgeAlpha', 0);
end

h_top = surf(x, y, z_top);
set(h_top, 'FaceAlpha', top_alpha, 'FaceColor', '#FF9BB3');
if show_grid
    set(h_top, 'EdgeAlpha', top_alpha*0.5, 'EdgeColor', [0 0 0]);
else
    set(h_top, 'EdgeAlpha', 0);
end

if show_arrow_bottom && any(bottom_mask(:))
    quiver3(x_sparse(bottom_mask), y_sparse(bottom_mask), z_bottom_sparse(bottom_mask), ...
        nx_bottom_sub(bottom_mask) .* vz_bottom_sub(bottom_mask), ...
        ny_bottom_sub(bottom_mask) .* vz_bottom_sub(bottom_mask), ...
        nz_bottom_sub(bottom_mask) .* vz_bottom_sub(bottom_mask), ...
        arrow_scale, 'Color', [0 0 1], 'LineWidth', arrow_line_width, ...
        'MaxHeadSize', arrow_head_size, 'AutoScale', 'off');
end

if show_arrow_top && any(top_mask(:))
    quiver3(x_sparse(top_mask), y_sparse(top_mask), z_top_sparse(top_mask), ...
        nx_top_sub(top_mask) .* vz_top_sub(top_mask), ...
        ny_top_sub(top_mask) .* vz_top_sub(top_mask), ...
        nz_top_sub(top_mask) .* vz_top_sub(top_mask), ...
        arrow_scale, 'Color', [1 0 0], 'LineWidth', arrow_line_width, ...
        'MaxHeadSize', arrow_head_size, 'AutoScale', 'off');
end

if show_arrow_middle && any(middle_mask(:))
    quiver3(x_sparse(middle_mask), y_sparse(middle_mask), z_middle_sparse(middle_mask), ...
        nx_middle_sub(middle_mask) .* vz_middle_sub(middle_mask), ...
        ny_middle_sub(middle_mask) .* vz_middle_sub(middle_mask), ...
        nz_middle_sub(middle_mask) .* vz_middle_sub(middle_mask), ...
        arrow_scale, 'Color', [0 0.5 0], 'LineWidth', arrow_line_width, ...
        'MaxHeadSize', arrow_head_size, 'AutoScale', 'off');
end

% 强制3D视图并设置初始视角
view(30, 30);  % 明确的3D初始视角（方位角30，仰角30）
lighting gouraud;
camlight('headlight');
xlabel('X');
ylabel('Y');
zlabel('Z');
axis equal;
hold off;

% 等待用户调整并确认
fprintf('提示：\n');
fprintf('1. 使用鼠标拖动可以旋转视图\n');
fprintf('2. 滚轮可以缩放视图\n');
fprintf('3. 按住Shift键拖动可以平移视图\n');
fprintf('4. 调整完成后，请按回车键继续生成视频...\n');
pause;

% 关键修改：明确获取并显示视角参数，确认是否为3D视角
[az, el] = view(gca);  % 分别获取方位角和仰角
fprintf('获取到的视角参数 - 方位角: %.2f, 仰角: %.2f\n', az, el);

% 检查是否接近平面视角，如果是则强制设置为3D视角
if abs(el) > 85 || abs(el) < -85 || abs(az) < 5
    warning('检测到接近平面视角，自动调整为3D视角');
    az = 30;  % 重置为合理的3D方位角
    el = 30;  % 重置为合理的3D仰角
end

axis_limits = axis(gca);  % 获取坐标轴范围
close(fig);

% ===== 生成视频部分 =====
videoFile = '3d_membrane_vector_field_final-3.avi';
writerObj = VideoWriter(videoFile, 'MPEG-4');
writerObj.FrameRate = 15;
open(writerObj);

fprintf('正在生成视频...\n');
for j = start_frame_idx:start_frame_idx+frames-1
    frame_idx = mod(j-1, frames) + 1;
    next_frame_idx = mod(j, frames) + 1;
    dt = 1 / writerObj.FrameRate;
    
    % 计算速度
    vz_bottom_raw = squeeze(z_bottom_all(next_frame_idx,:,:) - z_bottom_all(frame_idx,:,:)) / dt;
    vz_top_raw = squeeze(z_top_all(next_frame_idx,:,:) - z_top_all(frame_idx,:,:)) / dt;
    vz_middle_raw = squeeze(z_middle_all(next_frame_idx,:,:) - z_middle_all(frame_idx,:,:)) / dt;
    
    vz_bottom = max(min(vz_bottom_raw, max_arrow_speed), -max_arrow_speed);
    vz_top = max(min(vz_top_raw, max_arrow_speed), -max_arrow_speed);
    vz_middle = max(min(vz_middle_raw, max_arrow_speed), -max_arrow_speed);
    
    % 当前帧高度
    z_bottom = squeeze(z_bottom_all(frame_idx,:,:));
    z_top = squeeze(z_top_all(frame_idx,:,:));
    z_middle = squeeze(z_middle_all(frame_idx,:,:));
    
    % 后续绘图和视频写入代码与之前完全一致
    distance_from_center = sqrt(x.^2 + y.^2);
    pulse_influence = exp(-(distance_from_center/2).^2);
    
    [dx_bottom, dy_bottom] = gradient(z_bottom, 0.3, 0.3);
    normal_mag_bottom = sqrt(1 + dx_bottom.^2 + dy_bottom.^2);
    nx_bottom = -dx_bottom ./ normal_mag_bottom;
    ny_bottom = -dy_bottom ./ normal_mag_bottom;
    nz_bottom = 1 ./ normal_mag_bottom;
    
    [dx_top, dy_top] = gradient(z_top, 0.3, 0.3);
    normal_mag_top = sqrt(1 + dx_top.^2 + dy_top.^2);
    nx_top = -dx_top ./ normal_mag_top;
    ny_top = -dy_top ./ normal_mag_top;
    nz_top = 1 ./ normal_mag_top;
    
    [dx_middle, dy_middle] = gradient(z_middle, 0.3, 0.3);
    normal_mag_middle = sqrt(1 + dx_middle.^2 + dy_middle.^2);
    nx_middle = -dx_middle ./ normal_mag_middle;
    ny_middle = -dy_middle ./ normal_mag_middle;
    nz_middle = 1 ./ normal_mag_middle;
    
    % 绘图（与之前代码一致）
    fig = figure('Visible', 'off', 'Position', [100 100 fig_width fig_height]);
    set(fig, 'Renderer', 'OpenGL');  % 使用OpenGL渲染器确保3D效果
    hold on;
    
    h_middle = surf(x, y, z_middle);
    set(h_middle, 'FaceAlpha', middle_alpha, 'FaceColor', '#F5F5F5');
    if show_grid
        set(h_middle, 'EdgeAlpha', middle_alpha*0.5, 'EdgeColor', [0 0 0]);
    else
        set(h_middle, 'EdgeAlpha', 0);
    end
    
    h_bottom = surf(x, y, z_bottom);
    set(h_bottom, 'FaceAlpha', bottom_alpha, 'FaceColor', '#C9A0DC');
    if show_grid
        set(h_bottom, 'EdgeAlpha', bottom_alpha*0.5, 'EdgeColor', [0 0 0]);
    else
        set(h_bottom, 'EdgeAlpha', 0);
    end
    
    h_top = surf(x, y, z_top);
    set(h_top, 'FaceAlpha', top_alpha, 'FaceColor', '#FF9BB3');
    if show_grid
        set(h_top, 'EdgeAlpha', top_alpha*0.5, 'EdgeColor', [0 0 0]);
    else
        set(h_top, 'EdgeAlpha', 0);
    end
    
    % 绘制箭头（与之前代码一致）
    sparse_idx = 1:arrow_density:size(x,1);
    M = length(sparse_idx);
    
    x_sparse = x(sparse_idx, sparse_idx);
    y_sparse = y(sparse_idx, sparse_idx);
    z_bottom_sparse = z_bottom(sparse_idx, sparse_idx);
    z_top_sparse = z_top(sparse_idx, sparse_idx);
    z_middle_sparse = z_middle(sparse_idx, sparse_idx);
    
    nx_bottom_sub = nx_bottom(sparse_idx, sparse_idx);
    ny_bottom_sub = ny_bottom(sparse_idx, sparse_idx);
    nz_bottom_sub = nz_bottom(sparse_idx, sparse_idx);
    vz_bottom_sub = vz_bottom(sparse_idx, sparse_idx);
    
    nx_top_sub = nx_top(sparse_idx, sparse_idx);
    ny_top_sub = ny_top(sparse_idx, sparse_idx);
    nz_top_sub = nz_top(sparse_idx, sparse_idx);
    vz_top_sub = vz_top(sparse_idx, sparse_idx);
    
    nx_middle_sub = nx_middle(sparse_idx, sparse_idx);
    ny_middle_sub = ny_middle(sparse_idx, sparse_idx);
    nz_middle_sub = nz_middle(sparse_idx, sparse_idx);
    vz_middle_sub = vz_middle(sparse_idx, sparse_idx);
    
    pulse_influence_sub = pulse_influence(sparse_idx, sparse_idx);
    center_mask_bottom = pulse_influence_sub > center_region_threshold;
    center_mask_top = pulse_influence_sub > center_region_threshold;
    center_mask_middle = pulse_influence_sub > center_region_threshold;
    
    speed_bottom = sqrt(vz_bottom_sub.^2);
    speed_top = sqrt(vz_top_sub.^2);
    speed_middle = sqrt(vz_middle_sub.^2);
    speed_mask_bottom = (speed_bottom > speed_threshold) & (speed_bottom <= max_arrow_speed);
    speed_mask_top = (speed_top > speed_threshold) & (speed_top <= max_arrow_speed);
    speed_mask_middle = (speed_middle > speed_threshold) & (speed_middle <= max_arrow_speed);
    
    bottom_mask = center_mask_bottom & speed_mask_bottom;
    top_mask = center_mask_top & speed_mask_top;
    middle_mask = center_mask_middle & speed_mask_middle;
    
    if show_arrow_bottom && any(bottom_mask(:))
        quiver3(x_sparse(bottom_mask), y_sparse(bottom_mask), z_bottom_sparse(bottom_mask), ...
            nx_bottom_sub(bottom_mask) .* vz_bottom_sub(bottom_mask), ...
            ny_bottom_sub(bottom_mask) .* vz_bottom_sub(bottom_mask), ...
            nz_bottom_sub(bottom_mask) .* vz_bottom_sub(bottom_mask), ...
            arrow_scale, 'Color', [0 0 1], 'LineWidth', arrow_line_width, ...
            'MaxHeadSize', arrow_head_size, 'AutoScale', 'off');
    end
    
    if show_arrow_top && any(top_mask(:))
        quiver3(x_sparse(top_mask), y_sparse(top_mask), z_top_sparse(top_mask), ...
            nx_top_sub(top_mask) .* vz_top_sub(top_mask), ...
            ny_top_sub(top_mask) .* vz_top_sub(top_mask), ...
            nz_top_sub(top_mask) .* vz_top_sub(top_mask), ...
            arrow_scale, 'Color', [1 0 0], 'LineWidth', arrow_line_width, ...
            'MaxHeadSize', arrow_head_size, 'AutoScale', 'off');
    end
    
    if show_arrow_middle && any(middle_mask(:))
        quiver3(x_sparse(middle_mask), y_sparse(middle_mask), z_middle_sparse(middle_mask), ...
            nx_middle_sub(middle_mask) .* vz_middle_sub(middle_mask), ...
            ny_middle_sub(middle_mask) .* vz_middle_sub(middle_mask), ...
            nz_middle_sub(middle_mask) .* vz_middle_sub(middle_mask), ...
            arrow_scale, 'Color', [0 0.5 0], 'LineWidth', arrow_line_width, ...
            'MaxHeadSize', arrow_head_size, 'AutoScale', 'off');
    end
    
    % 应用视图参数
     axis(axis_limits);  % 应用保存的轴范围
    axis equal;         % 保持轴比例一致
    view(az, el);       % 应用用户调整的3D视角
    lighting gouraud;
    camlight('headlight');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');

    title(sprintf('三层膜系统的速度场 (帧 %d/%d)', frame_idx, frames));
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    drawnow;

    frame = getframe(fig);
    writeVideo(writerObj, frame);
    close(fig);
    
    fprintf('处理帧 %d/%d\r', frame_idx, frames);
end

close(writerObj);
fprintf('\n三层膜系统的视频已导出至: %s\n', fullfile(pwd, videoFile));