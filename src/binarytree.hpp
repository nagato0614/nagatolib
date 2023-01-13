#pragma once
//
// Created by nagato on 2023/01/01.
//

#ifndef NAGATOLIB_SRC_BINARYTREE_HPP_
#define NAGATOLIB_SRC_BINARYTREE_HPP_

#include <optional>
#include <memory>
#include <vector>

namespace nagato
{


/**
 * @brief 重複なしの二分探索木
 * @tparam T 等価比較可能な型
 */
template<typename T>
class BinaryTree
{

 public:
  struct BinaryTreeNode;

  // 型定義
  using NodeValue = std::unique_ptr<T>;
  using NodePtr = std::unique_ptr<BinaryTreeNode>;

  struct BinaryTreeNode
  {
	BinaryTreeNode(const T &v)
	{
	  value = std::unique_ptr<T>(new T);
	  *value = v;
	}
	NodeValue value;
	NodePtr left;
	NodePtr right;
  };

  explicit BinaryTree();

  explicit BinaryTree(const T &value);

  // コピーとムーブ禁止
  BinaryTree(const BinaryTree<T> &tree) = delete;
  BinaryTree(const BinaryTree<T> &&tree) = delete;

  /**
   * @brief 目的の値を探し, 値を追加する.
   * 先に左に追加しようとするがnullptrでなければ, 右に追加する.
   * 両方空いていない場合と見つからない場合falseが帰ってくる
   *
   * @param target_value 見つけたい値
   * @param add_value 追加する値
   * @return true 追加が成功した時
   * @return false 見つからない場合, 見つかっても子ノードが空いていない場合
   */
  bool find_and_add(T &target_value, T &add_value);

  NodePtr root;
};

template<typename T>
BinaryTree<T>::BinaryTree()
{
}

template<typename T>
BinaryTree<T>::BinaryTree(const T &value)
{
  root = std::make_unique<BinaryTreeNode>(value);
}

template<typename T>
bool BinaryTree<T>::find_and_add(T &target_value, T &add_value)
{
  static_assert(std::is_same<decltype(target_value == target_value),
							 bool>::value);

  std::vector<std::unique_ptr<T>&> nodes;
  nodes.push_back(root);
  while (nodes.size() > 0)
  {
	auto &node = nodes.front();

	if (node->value == target_value)
	{
	  // ターゲットと同じ値が見つかった場合
	  if (node->left)
	  {
		node->left = std::make_unique<T>(add_value);
	  }
	  else if (node->right)
	  {
		node->right = std::make_unique<T>(add_value);
	  }
	  else
	  {
		return false;
	  }
	}
	else
	{
	  // ターゲット違う場合は子ノードをリストに加える.
	  if (node->left)
	  {
		 nodes.push_back(node->left);
	  }

	  if (node->right)
	  {
		nodes.push_back(node->right);
	  }
	}

	// 取得したノードを削除
	nodes.erase(0);
  }

  return false;
}
}

#endif //NAGATOLIB_SRC_BINARYTREE_HPP_
