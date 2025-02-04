#include "KDTree.h"

#include "GeoConverter.hpp"


#if __APPLE__
int split_node;

kdtreegpu::kdtreegpu(int max_node_num, int max_neighbor_num, int query_num, int dim, sycl::queue& q_ct1, std::vector<vec3f>& points)
{

	q = q_ct1;
	kdtree_dim = dim;
	kdtree_max_neighbor_num = max_neighbor_num + 1;
	kdtree_query_num = query_num;
	kdtree_node_num = max_node_num;
	// 主机端分配内存
	n = sycl::malloc_host<Node>(kdtree_node_num, q_ct1);
	split = sycl::malloc_host<int>(kdtree_node_num, q_ct1);
	query_node = (float*)sycl::malloc_host(
		sizeof(float) * kdtree_dim * kdtree_query_num, q_ct1);
	query_result = (Pair_my*)sycl::malloc_host(
		sizeof(Pair_my) * kdtree_max_neighbor_num * kdtree_query_num,
		q_ct1);

	// 复制输入点数据到 n 数组
	//memcpy(n, points, sizeof(Node) * kdtree_node_num);
	n = new Node[kdtree_node_num];
	for (auto idx = 0; idx < kdtree_node_num; ++idx)
	{
		n[idx].point[0] = points[idx].x();
		n[idx].point[1] = points[idx].y();
		n[idx].point[2] = points[idx].z();
		n[idx].index = idx;
	}


}

//建树，采用非递归的形式，比递归的形式快
void kdtreegpu::build()
{

	std::pair<int, int> temp;

	temp.first = 0;
	temp.second = kdtree_node_num - 1;
	s.push(temp); // s为栈，用来存L, R, 相当于递归的栈

	while (!s.empty()) {

		temp = s.top();
		s.pop();

		int le = temp.first;
		int ri = temp.second;

		if (le > ri) {
			continue;
		}

		float var, maxs = -1;

		for (int i = 0; i < kdtree_dim; i++) {
			float ave = 0;
			for (int j = le; j <= ri; j++) {
				ave += n[j].point[i];
			}
			ave /= (ri - le + 1);
			var = 0;
			for (int j = le; j <= ri; j++) {
				var += (n[j].point[i] - ave) * (n[j].point[i] - ave);
			}
			var /= (ri - le + 1);
			if (var > maxs) {
				maxs = var;
				split_node = i;
			}
		}
		int mid = (le + ri) >> 1;
		split[mid] = split_node;
		std::nth_element(n + le, n + mid, n + ri + 1);
		s.push(std::make_pair(le, mid - 1));
		s.push(std::make_pair(mid + 1, ri));
	}
}

// gpu上并行查询
int kdtreegpu::query_gpu(float* query_points)
{
	std::vector<int> res;
	res.resize(361 * 181);

	// 复制查询点数据到 query_node 数组
	memcpy(query_node, query_points, sizeof(float) * kdtree_dim * kdtree_query_num);
	auto q_ct1 = q;
	
	Node* n_gpu = NULL;
	Pair_my* query_result_gpu;
	float* query_node_gpu = NULL;
	int* q_cur_id_gpu = NULL;
	int* split_gpu = NULL;

	//gpu上分配内存
	n_gpu = sycl::malloc_device<Node>((kdtree_node_num), q_ct1);
	query_result_gpu = (Pair_my*)sycl::malloc_device(
		sizeof(Pair_my) * (kdtree_max_neighbor_num)*kdtree_query_num,
		q_ct1);
	query_node_gpu = (float*)sycl::malloc_device(
		sizeof(float) * kdtree_dim * kdtree_query_num, q_ct1);
	q_cur_id_gpu = sycl::malloc_device<int>(kdtree_query_num, q_ct1);
	split_gpu = sycl::malloc_device<int>(kdtree_node_num, q_ct1);
	// 把数据从主机端拷贝到设备端
	q_ct1.memcpy((Node*)n_gpu, (Node*)n, sizeof(Node) * (kdtree_node_num));
	q_ct1.memcpy((int*)split_gpu, (int*)split, sizeof(int) * (kdtree_node_num));
	q_ct1.memcpy((float*)query_node_gpu, (float*)query_node,
		sizeof(float) * (kdtree_query_num * kdtree_dim));
	// 调用kernel函数，并行查询

	sycl::buffer<int, 1> res_buf(res.data(), sycl::range<1>(res.size()));

	q_ct1.submit([&](sycl::handler& cgh) {
		int kdtree_query_num_ct0 = kdtree_query_num;
		int kdtree_max_neighbor_num_ct6 = kdtree_max_neighbor_num;
		int kdtree_dim_ct7 = kdtree_dim;
		int kdtree_node_num_ct8 = kdtree_node_num;
		sycl::range<2> global_range(181, 361);
		sycl::range<2> local_range(16, 16);  // 选择一个合理的局部范围

		auto acc_res_buf = res_buf.get_access<sycl::access::mode::read_write>(cgh);

		cgh.parallel_for(
			sycl::nd_range<2>(global_range, local_range),
			[=](sycl::nd_item<2> item_ct1) {
				int height_index = item_ct1.get_global_id(0);
				int width_index = item_ct1.get_global_id(1);
				int global_id = height_index * 361 + width_index;

				vec2f current_pixel = { height_index, width_index };
				CartesianCoord current_position;
				SphericalCoord current_latlon_r;
				GeoConverter::convertPixelToLatLonToRadians(361, 181, -90.0, 90.0, -180.0, 180.0, current_pixel, current_latlon_r);
				GeoConverter::convertRadianLatLonToXYZ(current_latlon_r, current_position);

				query_node_gpu[0] = current_position.x();
				query_node_gpu[1] = current_position.y();
				query_node_gpu[2] = current_position.z();

				query_all_gpu(
					kdtree_query_num_ct0, split_gpu, q_cur_id_gpu,
					query_node_gpu, query_result_gpu, n_gpu,
					kdtree_max_neighbor_num_ct6, kdtree_dim_ct7,
					kdtree_node_num_ct8, item_ct1);


				auto nearest_node = query_result_gpu[1];
				auto result_idx = nearest_node.id;
				int cell_id = n_gpu[result_idx].index;
				acc_res_buf[global_id] = cell_id;
			});
		});
	//同步，所有的线程执行完
	q_ct1.wait_and_throw();
	//将结果拷贝回主机端
	q_ct1
		.memcpy(query_result, query_result_gpu,
			sizeof(Pair_my) * kdtree_max_neighbor_num *
			kdtree_query_num)
		.wait();
	//释放设备端显存
	sycl::free(n_gpu, q_ct1);
	sycl::free(query_result_gpu, q_ct1);
	sycl::free(split_gpu, q_ct1);
	sycl::free(query_node_gpu, q_ct1);
	sycl::free(q_cur_id_gpu, q_ct1);

	std::ofstream outfile("output22222.txt");

	if (outfile.is_open()) {
		for (size_t i = 0; i < 181; ++i) {
			for (size_t j = 0; j < 361; ++j) {
				outfile << res[i * 361 + j] << " ";
			}
			outfile << "\n";  // New line at the end of each row
		}
		outfile.close();
		std::cout << "Data has been written to output.txt" << std::endl;
	}
	else {
		std::cerr << "Failed to open the file for writing!" << std::endl;
		return 1;
	}


	auto idx = query_result[1].id;
	return n[idx].index;
}

//cpu上查询一个点
void kdtreegpu::query_one(int left, int right, int id)
{
	Stack_my sta[20], temp;

	int cou = 0;
	sta[0].first = left;
	sta[0].second = right;
	cou++;
	while (cou) {

		temp = sta[cou - 1];
		cou--;

		if (temp.first > temp.second) {
			continue;
		}
		if (que.size() == kdtree_max_neighbor_num - 1 && temp.val > que.top().dis) {
			continue;
		}

		int mid = (temp.first + temp.second) >> 1;
		int cur = split[mid];
		float ans = 0;
		for (int i = 0; i < kdtree_dim; i++)
		{
			ans += (query_node[id * kdtree_dim + i] - n[mid].point[i]) *
				(query_node[id * kdtree_dim + i] - n[mid].point[i]);
		}
		if (cnt < kdtree_max_neighbor_num - 1)
		{
			Pair_my tmp;
			tmp.id = mid;
			tmp.dis = ans;
			que.push(tmp);
			cnt++;
		}
		else if (ans < que.top().dis)
		{
			Pair_my tmp;
			tmp.id = mid;
			tmp.dis = ans;
			que.pop();
			que.push(tmp);
		}
		float radiu = abs(query_node[id * kdtree_dim + cur] - n[mid].point[cur]);
		if (query_node[id * kdtree_dim + cur] < n[mid].point[cur]) {

			sta[cou].first = mid + 1;
			sta[cou].second = temp.second;
			sta[cou].val = radiu;
			cou++;

			sta[cou].first = temp.first;
			sta[cou].second = mid - 1;
			sta[cou].val = 0;
			cou++;
		}
		else {
			sta[cou].first = temp.first;
			sta[cou].second = mid - 1;
			sta[cou].val = radiu;
			cou++;
			sta[cou].first = mid + 1;
			sta[cou].second = temp.second;
			sta[cou].val = 0;
			cou++;
		}
	}

}

// cpu上查询，并验证cpu的查询和gpu的查询结果是否一致
int kdtreegpu::query_cpu_and_check()
{
	int error = 0;
	for (int i = 0; i < kdtree_query_num; i++)
	{
		cnt = 0;
		query_one(0, kdtree_node_num - 1, i);

		int count = 0;
		int num = que.size();
		while (!que.empty())
		{
			Pair_my tmp;
			tmp.id = que.top().id;
			tmp.dis = que.top().dis;

			for (int k = 1; k < kdtree_max_neighbor_num; k++)
			{
				if (query_result[i * kdtree_max_neighbor_num + k].id == tmp.id)
				{
					count++;
				}
			}

			que.pop();
		}

		if (count != num)
		{
			printf("%d %d\n", i, count);
			printf("EROOR\n");
			error++;
		}
	}
	return error;
}

kdtreegpu::~kdtreegpu() 
{
	sycl::free(query_node, q);
	sycl::free(split, q);
	sycl::free(query_result, q);
}



void swap_gpu(int id, int id1, int id2, Pair_my* query_result_gpu, int kdtree_max_neighbor_num)
{
	Pair_my temp;

	temp.id = query_result_gpu[id * kdtree_max_neighbor_num + id1].id;
	temp.dis = query_result_gpu[id * kdtree_max_neighbor_num + id1].dis;

	query_result_gpu[id * kdtree_max_neighbor_num + id1].id = query_result_gpu[id * kdtree_max_neighbor_num + id2].id;
	query_result_gpu[id * kdtree_max_neighbor_num + id1].dis = query_result_gpu[id * kdtree_max_neighbor_num + id2].dis;
	query_result_gpu[id * kdtree_max_neighbor_num + id2].id = temp.id;
	query_result_gpu[id * kdtree_max_neighbor_num + id2].dis = temp.dis;

}

void push_gpu(float dis_, int id_, int id, int* q_cur_id_gpu, Pair_my* query_result_gpu, int kdtree_max_neighbor_num)
{
	query_result_gpu[id * kdtree_max_neighbor_num + q_cur_id_gpu[id]].id = id_;
	query_result_gpu[id * kdtree_max_neighbor_num + q_cur_id_gpu[id]].dis = dis_;

	int re = q_cur_id_gpu[id];

	while (re > 1)
	{

		if (query_result_gpu[id * kdtree_max_neighbor_num + re].dis > query_result_gpu[id * kdtree_max_neighbor_num + (re >> 1)].dis)
		{
			swap_gpu(id, re, re >> 1, query_result_gpu, kdtree_max_neighbor_num);
			re >>= 1;
		}
		else
		{
			break;
		}
	}
	q_cur_id_gpu[id]++;
}


void pop_gpu(int id, int* q_cur_id_gpu, Pair_my* query_result_gpu, int kdtree_max_neighbor_num)
{

	query_result_gpu[id * kdtree_max_neighbor_num + 1] = query_result_gpu[id * kdtree_max_neighbor_num + q_cur_id_gpu[id] - 1];
	q_cur_id_gpu[id]--;

	int re = 1;
	while (re < q_cur_id_gpu[id])
	{
		if ((re << 1) < q_cur_id_gpu[id])
		{

			if ((re << 1 | 1) < q_cur_id_gpu[id])
			{
				if (query_result_gpu[id * kdtree_max_neighbor_num + (re << 1)].dis >= query_result_gpu[id * kdtree_max_neighbor_num + (re << 1 | 1)].dis)
				{

					if (query_result_gpu[id * kdtree_max_neighbor_num + (re << 1)].dis > query_result_gpu[id * kdtree_max_neighbor_num + re].dis)
					{
						swap_gpu(id, re << 1, re, query_result_gpu, kdtree_max_neighbor_num);
						re <<= 1;
					}
					else
					{
						break;
					}
				}
				else
				{
					if (query_result_gpu[id * kdtree_max_neighbor_num + (re << 1 | 1)].dis > query_result_gpu[id * kdtree_max_neighbor_num + re].dis)
					{
						swap_gpu(id, re << 1 | 1, re, query_result_gpu, kdtree_max_neighbor_num);
						re <<= 1;
						re |= 1;
					}
					else
					{
						break;
					}
				}
			}
			else
			{
				if (query_result_gpu[id * kdtree_max_neighbor_num + (re << 1)].dis > query_result_gpu[id * kdtree_max_neighbor_num + re].dis)
				{
					swap_gpu(id, re << 1, re, query_result_gpu, kdtree_max_neighbor_num);
					re <<= 1;
				}
				else
				{
					break;
				}
			}
		}
		else
		{
			break;
		}
	}
}

void query_one_gpu(int left, int right, int idx, int* split_gpu,
	float* query_node_gpu, Pair_my* query_result_gpu,
	Node* n_gpu, int* q_cur_id_gpu, int kdtree_max_neighbor_num,
	int kdtree_dim)
{
	Stack_my sta[20], temp;
	int cou = 0;
	sta[0].first = left;
	sta[0].second = right;
	cou++;
	while (cou) {

		temp = sta[cou - 1];
		cou--;
		if (temp.first > temp.second) {
			continue;
		}
		if (q_cur_id_gpu[idx] == kdtree_max_neighbor_num && temp.val >
			query_result_gpu[idx * kdtree_max_neighbor_num + 1].dis) {
			continue;
		}

		int mid = (temp.first + temp.second) >> 1;
		int cur = split_gpu[mid];
		float ans = 0;
		for (int i = 0; i < kdtree_dim; i++)
		{
			ans += (query_node_gpu[idx * kdtree_dim + i] -
				n_gpu[mid].point[i]) *
				(query_node_gpu[idx * kdtree_dim + i] -
					n_gpu[mid].point[i]);
		}
		if (q_cur_id_gpu[idx] < kdtree_max_neighbor_num)
		{
			push_gpu(ans, mid, idx, q_cur_id_gpu, query_result_gpu, kdtree_max_neighbor_num);
		}
		else if (ans < query_result_gpu[idx * kdtree_max_neighbor_num + 1].dis)
		{
			pop_gpu(idx, q_cur_id_gpu, query_result_gpu, kdtree_max_neighbor_num);
			push_gpu(ans, mid, idx, q_cur_id_gpu, query_result_gpu, kdtree_max_neighbor_num);
		}

		float radiu = abs(query_node_gpu[idx * kdtree_dim + cur]
			- n_gpu[mid].point[cur]);

		if (query_node_gpu[idx * kdtree_dim + cur]
			< n_gpu[mid].point[cur]) {

			sta[cou].first = mid + 1;
			sta[cou].second = temp.second;
			sta[cou].val = radiu;
			cou++;

			sta[cou].first = temp.first;
			sta[cou].second = mid - 1;
			sta[cou].val = 0;
			cou++;
		}
		else {

			sta[cou].first = temp.first;
			sta[cou].second = mid - 1;
			sta[cou].val = radiu;
			cou++;

			sta[cou].first = mid + 1;
			sta[cou].second = temp.second;
			sta[cou].val = 0;
			cou++;
		}
	}
}

void query_all_gpu(int query_num, int* split_gpu,
	int* q_cur_id_gpu, float* query_node_gpu,
	Pair_my* query_result_gpu,
	Node* n_gpu, int kdtree_max_neighbor_num,
	int kdtree_dim, int kdtree_node_num, const sycl::nd_item<2>& item_ct1)
{

	// 获取全局ID
	int global_id_x = item_ct1.get_global_id(1);
	int global_id_y = item_ct1.get_global_id(0);

	// 计算全局索引，假设每个工作项对应一个query
	int i = global_id_y * item_ct1.get_global_range(1) + global_id_x;
	// 确保索引不超出范围
	if (i < query_num) {
		q_cur_id_gpu[i] = 1;
		query_one_gpu(0, kdtree_node_num - 1, i,
			split_gpu, query_node_gpu,
			query_result_gpu, n_gpu, q_cur_id_gpu,
			kdtree_max_neighbor_num, kdtree_dim);
	}
	/*for (int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
		item_ct1.get_local_id(2);
		i < query_num;
		i += item_ct1.get_local_range(2) * item_ct1.get_group_range(2))
	{
		q_cur_id_gpu[i] = 1;
		query_one_gpu(0, kdtree_node_num - 1, i,
			split_gpu, query_node_gpu,
			query_result_gpu, n_gpu, q_cur_id_gpu,
			kdtree_max_neighbor_num, kdtree_dim);
	}*/
}






#endif