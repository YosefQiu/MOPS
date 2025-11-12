#pragma once
#include "ggl.h"
#include "Utils/nanoflann.hpp"

namespace MOPS
{
    #if _WIN32 || __linux__
    template <
        class VectorOfVectorsType, typename num_t = double, int DIM = -1,
        class Distance = nanoflann::metric_L2, typename IndexType = size_t>
        struct KDTreeVectorOfVectorsAdaptor
    {
        using self_t = KDTreeVectorOfVectorsAdaptor<
            VectorOfVectorsType, num_t, DIM, Distance, IndexType>;
        using metric_t =
            typename Distance::template traits<num_t, self_t>::distance_t;
        using index_t =
            nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM, IndexType>;

        /** The kd-tree index for the user to call its methods as usual with any
        * other FLANN index */
        index_t* index = nullptr;
        KDTreeVectorOfVectorsAdaptor() = default;
        /// Constructor: takes a const ref to the vector of vectors object with the
        /// data points
        KDTreeVectorOfVectorsAdaptor(
            const size_t /* dimensionality */, const VectorOfVectorsType& mat,
            const int leaf_max_size = 10, const unsigned int n_thread_build = 1, bool bInitIndex = true)
            : m_data(mat)
        {
            assert(mat.size() != 0/* && mat[0].size() != 0*/);
            const size_t dims = 3 /*mat[0].size()*/;
            if (DIM > 0 && static_cast<int>(dims) != DIM)
                throw std::runtime_error(
                    "Data set dimensionality does not match the 'DIM' template "
                    "argument");
            if (bInitIndex)
            {
                index = new index_t(
                    static_cast<int>(dims), *this /* adaptor */,
                    nanoflann::KDTreeSingleIndexAdaptorParams(
                        leaf_max_size, nanoflann::KDTreeSingleIndexAdaptorFlags::None,
                        n_thread_build));
            }
            else
            {
                index = new index_t(
                    static_cast<int>(dims), *this /* adaptor */,
                    nanoflann::KDTreeSingleIndexAdaptorParams(
                        leaf_max_size, nanoflann::KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex,
                        n_thread_build));
            }
            
        }

        ~KDTreeVectorOfVectorsAdaptor() { delete index; }

        const VectorOfVectorsType& m_data;

        inline void query(
            const num_t* query_point, const size_t num_closest,
            IndexType* out_indices, num_t* out_distances_sq) const
        {
            nanoflann::KNNResultSet<num_t, IndexType> resultSet(num_closest);
            resultSet.init(out_indices, out_distances_sq);
            index->findNeighbors(resultSet, query_point);
        }


        const self_t& derived() const { return *this; }
        self_t& derived() { return *this; }

        inline size_t kdtree_get_point_count() const { return m_data.size(); }

        inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const
        {
            return m_data[idx][dim];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /*bb*/) const
        {
            return false;
        }



    };

    typedef std::vector<vec3> my_vector_of_vectors_t;
    typedef KDTreeVectorOfVectorsAdaptor<my_vector_of_vectors_t, double> KDTree_t;
    #elif __APPLE__



    extern int split_node;
    // KDTree node
    struct Node {

        float point[3];
        int index;
        bool operator<(const Node& n)const {
            return point[split_node] < n.point[split_node];
        }

    };
    struct Pair_my {
        int id;
        float dis;

        bool operator<(const Pair_my& p)const {
            return dis < p.dis;
        }
    };

    // 
    struct Stack_my
    {
        int first;
        int second;
        float val;
    };


    class kdtreegpu {

    public:

        kdtreegpu(int max_node_num, int max_neighbor_num, int query_num, int dim, sycl::queue& q_ct1, std::vector<vec3f>& points); 
        void build(); 
        int query_gpu(float* query_points); 
        void query_one(int left, int right, int id); 
        int query_cpu_and_check(); 
        virtual ~kdtreegpu();
    public:
        int kdtree_dim; 
        int kdtree_max_neighbor_num; 
        int kdtree_query_num; 
        int kdtree_node_num; 
        int cnt;

        int* split; 
        Node* n;  
        Pair_my* query_result; 
        float* query_node; 

        std::stack<std::pair<int, int> > s; 
        std::priority_queue< Pair_my > que; 
        sycl::queue q;
    };


    void swap_gpu(int id, int id1, int id2, Pair_my* query_result_gpu, int kdtree_max_neighbor_num);



    void push_gpu(float dis_, int id_, int id, int* q_cur_id_gpu, Pair_my* query_result_gpu, int kdtree_max_neighbor_num);

    void pop_gpu(int id, int* q_cur_id_gpu, Pair_my* query_result_gpu, int kdtree_max_neighbor_num);

  
    SYCL_EXTERNAL void query_one_gpu(int left, int right, int idx, int* split_gpu,
        float* query_node_gpu, Pair_my* query_result_gpu, Node* n_gpu, int* q_cur_id_gpu, int kdtree_max_neighbor_num, int kdtree_dim);

 
    SYCL_EXTERNAL void query_all_gpu(int query_num, int* split_gpu,
        int* q_cur_id_gpu, float* query_node_gpu,
        Pair_my* query_result_gpu,
        Node* n_gpu, int kdtree_max_neighbor_num,
        int kdtree_dim, int kdtree_node_num, const sycl::nd_item<2>& item_ct1);

    #endif
}