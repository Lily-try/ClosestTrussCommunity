/*
 * list.h
 *
 *  Created on: 9May,2017
 *      Author: longyuan
 */

#ifndef LIST_H_
#define LIST_H_

struct LinkedList;

struct Link {
	void * data;
	Link* next;
	Link* prev;
	LinkedList* linkedList;
};

struct LinkedList {
	Link* head;
	Link* tail;
	int length;
};

int isHead(Link* list);

int isTail(Link* list);

void* deleteNode(Link* list);

Link* addAfter(Link* list, void* data);

Link* addBefore(Link* list, void* data);

void addLinkBefore(Link* list, Link* newLink);

Link* removeLink(Link* list);

LinkedList* createLinkedList(void);

void destroyLinkedList(LinkedList* linkedList);

void copyLinkedList(LinkedList* destination, LinkedList* source);

int contains(LinkedList* linkedList, void* data,
		int (*comparator)(void*, void*));

int equal(LinkedList* list1, LinkedList* list2,
		int (*comparator)(void*, void*));

void restoreLinksWithReferences(LinkedList* list);

Link* addFirst(LinkedList* linkedList, void* data);

Link* addLast(LinkedList* linkedList, void* data);

Link* removeFirst(LinkedList* linkedList);

Link* removeLast(LinkedList* linkedList);

void deleteLast(LinkedList* linkedList);

void* getFirst(LinkedList* linkedList);

void printListAbbv(LinkedList* linkedList, void (*printFunc)(void*));

void printList(LinkedList* linkedList, void (*printFunc)(void*));

int length(LinkedList* linkedList);

int isEmpty(LinkedList* linkedList);

#endif /* LIST_H_ */
